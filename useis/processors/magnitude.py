from ..core.project_manager import ProjectManager
from uquake.core.stream import Stream
from uquake.core.event import Event
import numpy as np
from uquake.waveform.simple_mag import moment_magnitude
from pathlib import Path
import os
import shutil


class Magnitude(ProjectManager):
    def __init__(self, base_projects_path: Path, project_name: str,
                 network_code: str):
        """
        Based object for the magnitude calculation
        :param base_projects_path: base project path
        :type base_projects_path: str
        :param project_name: project name or id
        :type project_name: str
        :param network_code: network name or id
        :type network_code: str
        """

        super().__init__(base_projects_path, project_name, network_code)

        self.files.magnitude_settings = self.paths.config / \
                                        'magnitude_settings.toml'

        if not self.files.magnitude_settings.is_file():
            settings_template = Path(os.path.realpath(__file__)).parent / \
                                '../settings/magnitude_settings_template.toml'

            shutil.copyfile(settings_template,
                            self.files.magnitude_settings)

            super().__init__(base_projects_path, project_name, network_code)


class MwPE(Magnitude):
    def __init__(self, base_projects_path: Path, project_name: str,
                 network_code: str):
        """
        Based object for the magnitude calculation
        :param base_projects_path: base project path
        :type base_projects_path: str
        :param project_name: project name or id
        :type project_name: str
        :param network_code: network name or id
        :type network_code: str
        """

        super().__init__(base_projects_path, project_name, network_code)

    def predict(self, st: Stream, event_location: np.array,
                use_distance_along_rays=True):
        """

        :param st: waveform
        :type st: :py:class:uquake.core.stream.Stream:
        :param event_location: event location
        :type event_location: :py:class:numpy.array:
        :param use_distance_along_rays: Use the distance along the rays if True
        use the straight line distance if False. Default True.
        :type use_distance_along_rays: bool
        :return: the magnitude value estimate
        :rtype: float
        """

        a = self.settings.MwPE.a
        b = self.settings.MwPE.c

        min_frequency = 0
        for tr in st:
            trace_time = (tr.stats.endtime - tr.stats.starttime) * 0.9
            if 1 / trace_time > min_frequency:
                min_frequency = 1 / trace_time

        min_frequency_1 = min_frequency
        min_frequency_2 = 10 ** (np.log10(min_frequency) + 0.2)

        pre_filt = [min_frequency_1, min_frequency_2]

        max_frequency_2 = self.settings.MwPE.max_frequency
        max_frequency_1 = 10 ** (np.log10(max_frequency_2) - 0.2)

        pre_filt = [min_frequency_1, min_frequency_2,
                    max_frequency_1, max_frequency_2]

        water_level = self.settings.MwPE.water_level

        st.attach_response(self.inventory)
        st.remove_response(inventory=self.inventory, pre_filt=pre_filt,
                           water_level=water_level, output='VEL')

        if use_distance_along_rays:
            rays = self.travel_times.ray_tracer(event_location,
                                                seed_labels='current_event')

        magnitudes = []
        for network in self.inventory:
            for station in network:
                st2 = st.copy().select(network=network, station=station)
                if len(st2) == 1:
                    amplitude = np.max(st2[0].data) * np.sqrt(3)
                else:
                    amplitude = np.max(st2.composite()[0].data)

        networks = self.inventory


class MomentMagnitude(Magnitude):
    def __init__(self, base_projects_path: Path, project_name: str,
                 network_code: str):
        """
        Based object for the magnitude calculation
        :param base_projects_path: base project path
        :type base_projects_path: str
        :param project_name: project name or id
        :type project_name: str
        :param network_code: network name or id
        :type network_code: str
        """

        super().__init__(base_projects_path, project_name, network_code)

    def predict(self, st: Stream, event: Event, only_triaxial=True,
                density=2700, min_dist=20, win_length=0.04, len_spectrum=2**12,
                clipped_fraction=0.1, max_frequency=600,
                preferred_origin_only=True):
        """
        Calculate the moment magnitude for a given event
        :param st: waveform
        :type st: :py:class:uquake.core.stream.Stream:
        :param event: event location
        :type event: :py:class:numpy.array:
        :return: the magnitude value estimate
        :rtype: float
        :param only_triaxial: whether only triaxial sensor are used in the
        magnitude calculation (optional) (not yet implemented)
        :type only_triaxial: bool
        :param density: density in kg / m**3 (assuming homogeneous for now)
        :type density: float
        :param win_length: length of the window in second in which magnitude is
        calculated
        :type win_length: float
        :param min_dist: minimum distance between sensor an event to allow
        magnitude calculation
        :param len_spectrum: length of the spectrum
        :param clipped_fraction: allowed clipped fraction (fraction of the
        signal equal to the min or the max.
        :param max_frequency: maximum frequency used in the calculation on
        magnitude. After a certain frequency, the noise starts to dominate the
        signal and the biases the calculation of the magnitude and corner
        frequency.
        :param preferred_origin_only: calculates the magnitude for the
        preferred_origin only
        :rtype: uquake.core.event.Catalog
        """

        mag = moment_magnitude(st, event, inventory, self.p_velocity,
                               self.s_velocity, only_triaxial=only_triaxial,
                               density=density, min_dist=min_dist,
                               win_length=win_length,
                               len_spectrum=len_spectrum,
                               clipped_fraction=clipped_fraction,
                               max_frequency=max_frequency,
                               preferred_origin_only=preferred_origin_only)

        return mag





            


