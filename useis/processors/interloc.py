from ..core.project_manager import ProjectManager
from uquake.core.stream import Stream
from uquake.core.logging import logger
from pathlib import Path
import os
import shutil
import numpy as np
from time import time
from uquake.core.util import tools
from xseis2 import xspy


class Interloc(ProjectManager):
    def __init__(self, base_projects_path, project_name, network_code,
                 **kwargs):
        
        super().__init__(base_projects_path, project_name, network_code, 
                         **kwargs)

        self.files.magnitude_settings = self.paths.config / \
                                        'interloc_settings.toml'

        if not self.files.magnitude_settings.is_file():
            settings_template = Path(os.path.realpath(__file__)).parent / \
                                '../settings/interloc_settings_template.toml'

            shutil.copyfile(settings_template,
                            self.files.magnitude_settings)

            super().__init__(base_projects_path, project_name, network_code,
                             **kwargs)
        
    def locate(self, stream: Stream):

        logger.info("pipeline: interloc")

        # TODO: copy not necessary test application is broken

        nthreads = self.settings.interloc.nthreads
        fixed_wlen_sec = self.settings.interloc.fixed_wlen_sec
        sample_rate_decimated = self.settings.interloc.samplerate_decimated
        pair_dist_min = self.settings.interloc.pair_dist_min
        pair_dist_max = self.settings.interloc.pair_dist_max
        cc_smooth_length_sec = self.settings.interloc.cc_smooth_length_sec

        whiten_corner_freqs = np.array(
            self.settings.interloc.whiten_corner_freqs, dtype=np.float32)

        stalocs = self.htt.locations
        ttable = (self.htt.hf["ttp"][:] * sample_rate_decimated).astype(
            np.uint16)
        ttable_s = (self.htt.hf["tts"][:] * sample_rate_decimated).astype(
            np.uint16)
        ngrid = ttable.shape[1]
        ttable_row_ptrs = np.array(
            [row.__array_interface__["data"][0] for row in ttable])
        ttable_row_ptrs_s = np.array(
            [row.__array_interface__["data"][0] for row in ttable_s])

        logger.info("preparing data for Interloc")
        t4 = time()

        # remove channels which do not have matching ttable entries
        # This should be handled upstream

        for trace in stream:
            station = trace.stats.station
            component = trace.stats.channel
            if trace.stats.site not in self.htt.sites:
                logger.warning(f'Sensor {station} not in the H5 travel time '
                               f'file sensor list... removing trace for '
                               f'{station} and component {component}')
                stream.remove(trace)
            elif np.max(trace.data) == 0:
                # from ipdb import set_trace; set_trace()
                logger.warning(f'trace for component {component} of sensor'
                               f' {station} contains only zero... removing '
                               f'trace')
                stream.remove(trace)
            elif trace.stats.site in self.settings.sites.black_list:
                logger.warning(f'sensor {station} is in the black list... '
                               f'removing trace for sensor {station} and '
                               f'component {component}')

                stream.remove(trace)

        sample_rate = stream[0].stats.sampling_rate
        decimate_factor = int(sample_rate / sample_rate_decimated)
        if decimate_factor == 0:
            stream = stream.resample(sample_rate_decimated)
            decimate_factor = 1

        data, sample_rate, t0 = stream.as_array(fixed_wlen_sec)
        data = np.nan_to_num(data)
        decimate_factor = int(sample_rate / sample_rate_decimated)
        if decimate_factor > 1:
            data = tools.decimate(data, sample_rate, decimate_factor)
        channel_map = stream.channel_map().astype(np.uint16)

        ikeep = self.htt.index_sites(stream.unique_sites)
        debug_file = self.paths.debug / f'iloc_{str(t0)}.npz'
        t5 = time()
        logger.info(
            "done preparing data for Interloc in %0.3f seconds" % (t5 - t4))

        debug_level = self.settings.interloc.debug_level
        debug_level = 0

        logger.info("Locating event with Interloc")
        t6 = time()
        logger.info(
            "sample_rate_decimated {}, ngrid {}, nthreads {}, debug {}, "
            "debug_file {}",
            sample_rate_decimated, ngrid, nthreads,
            debug_level,
            debug_file)

        from ipdb import set_trace
        set_trace()
        out = xspy.pySearchOnePhase(
            data,
            int(sample_rate_decimated),
            channel_map,
            stalocs[ikeep],
            ttable_row_ptrs[ikeep],
            ngrid,
            whiten_corner_freqs,
            pair_dist_min,
            pair_dist_max,
            cc_smooth_length_sec,
            nthreads,
            debug_level,
            str(debug_file)
        )

        out_s = xspy.pySearchOnePhase(
            data,
            int(sample_rate_decimated),
            channel_map,
            stalocs[ikeep],
            ttable_row_ptrs_s[ikeep],
            ngrid,
            whiten_corner_freqs,
            pair_dist_min,
            pair_dist_max,
            cc_smooth_length_sec,
            nthreads,
            debug_level,
            str(debug_file)
        )

        vmax, imax, iot = out
        vmax_s, imax_s, iot_s = out_s

        if vmax_s > vmax:
            vmax = vmax_s
            imax = imax_s
            iot = iot_s
            logger.info('stacking along the s-wave moveout curve yielded '
                        'better result')

        normed_vmax = vmax * fixed_wlen_sec
        lmax = self.htt.icol_to_xyz(imax)
        t7 = time()
        logger.info("Done locating event with Interloc in %0.3f seconds" % (
                t7 - t6))

        t0_epoch = t0.timestamp
        ot_epoch = t0_epoch + iot / sample_rate_decimated

        method = "%s" % ("INTERLOC",)

        logger.info("power: %.3f, ix_grid: %d, ix_ot: %d" % (vmax, imax, iot))
        logger.info("utm_loc: {}", lmax)
        logger.info("=======================================\n")
        logger.info("VMAX over threshold (%.3f)" % (vmax))

        self.response = {'x': lmax[0],
                         'y': lmax[1],
                         'z': lmax[2],
                         'vmax': vmax,
                         'normed_vmax': normed_vmax,
                         'event_time': ot_epoch,
                         'method': method}

        return self.response