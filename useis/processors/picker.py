import numpy as np
import pandas as pd
from ..core.project_manager import ProjectManager
from pathlib import Path
import os
import shutil
from ..settings.settings import Settings
from uquake.waveform.pick import snr_ensemble_re_picker, calculate_snr
from uquake.core.event import (Pick, WaveformStreamID, ResourceIdentifier,
                               Arrival, Origin, Event)
from uquake.core import UTCDateTime
import uquake
from ..ai.model import AIPicker
from datetime import datetime
from uquake.core.stream import Stream
from uquake.core.inventory import Inventory
from typing import List
from obspy.realtime.signal import kurtosis
from numpy.fft import fftshift
from loguru import logger
from typing import List


class PickerResult(object):
    def __init__(self, new_picks, initial_picks, snr_threshold, stream):
        self.new_picks = new_picks
        self.initial_picks = initial_picks
        self.snr_threshold = snr_threshold
        self.stream = stream

        for k, pick in enumerate(self.new_picks):
            network = pick.waveform_id.network_code
            station = pick.waveform_id.station_code
            location = pick.waveform_id.location_code

            snr = calculate_snr(self.stream.select(network=network,
                                                   station=station,
                                                   location=location
                                                   )[0],
                                pick.time,
                                pre_wl=0.05, post_wl=0.02)

            self.new_picks[k].snr = snr

    def export_event(self):
        return self.append_event()

    @property
    def picks(self):
        picks = []
        for pick in self.new_picks:
            if self.snr_threshold is not None:
                if pick.snr < self.snr_threshold:
                    continue
            picks.append(pick)

        return picks

    def append_event(self, event: Event = None):

        snr_threshold = self.snr_threshold
        arrivals = []
        picks = []

        for pick in self.new_picks:
            picks.append(pick)

        for pick in self.initial_picks:
            picks.append(pick)

            arrival = Arrival(pick_id=pick.resource_id,
                              phase=pick.phase_hint)
            arrivals.append(arrival)

            origin = Origin(arrivals=arrivals)

        if event is None:
            event = Event(picks=picks, origins=[origin])

            return event

        if isinstance(event, uquake.core.event.Catalog):
            event[0].origins.append(origin)
            for pick in self.new_picks:
                event[0].picks.append(pick)

            return event

        else:
            event.origins.append(origin)
            for pick in self.new_picks:
                event.picks.append(pick)

            return event

    # def measure_incidence(self, stream: Stream,
    #                       inventory: Inventory):
    #
    #     pass

    def write_simple_pick_file(self, filename):
        with open(filename, 'w') as pick_file_out:
            for pick in self.picks:
                network = pick.waveform_id.network_code
                station = pick.waveform_id.station_code
                location = pick.waveform_id.location_code
                channel = None
                if pick.waveform_id.channel_code is not None:
                    channel = pick.waveform_id.channel_code[0:2]

                if self.snr_threshold is not None:
                    if pick.snr < self.snr_threshold:
                        continue

                station_string = f'{network}.{station}.'
                # {location}.{channel}*'
                line_out = f'{station_string},{pick.phase_hint.upper()},' \
                           f'{pick.time.timestamp}\n'
                pick_file_out.write(line_out)


class Picker(ProjectManager):

    def __init__(self, base_projects_path: Path, project_name: str,
                 network_code: str):
        """
        Object to manage repicking within a project
        :param base_projects_path: base project path
        :type base_projects_path: str
        :param project_name: project name or id
        :type project_name: str
        :param network_code: network name or id
        :type network_code: str
        """

        super().__init__(base_projects_path, project_name, network_code)
        self.files.picker_settings = self.paths.config / 'picker_settings.toml'

        if not self.files.picker_settings.is_file():
            settings_template = Path(os.path.realpath(__file__)).parent / \
                            '../settings/picker_settings_template.toml'

            shutil.copyfile(settings_template,
                            self.files.picker_settings)
            super().__init__(base_projects_path, project_name, network_code)

        self.files.picker_model = self.paths.ai_models / 'picker.pickle'

        self.ai_picker = None

        if self.files.picker_model.is_file():
            if self.files.picker_model.is_file():
                self.ai_picker = AIPicker.read(
                    self.files.picker_model)

    @staticmethod
    def read_nmx_pick(pick_file: str):

        nmx_picks = pd.read_csv(pick_file,
                                names=['site_id', 'phase', 'timestamp'])

        nmx_picks['datetime'] = [datetime.utcfromtimestamp(
            pick[1]['timestamp']) for pick in nmx_picks.iterrows()]
        out_picks = []
        for pick in nmx_picks.iterrows():
            pick = pick[1]
            if len(pick.site_id.split('.')) == 3:
                network, station, location = pick.site_id.split('.')
                channel = None
            elif len(pick.site_id.split('.')) == 4:
                network, station, location, channel = pick.site_id.split('.')

            waveform_id = WaveformStreamID(network_code=network,
                                           station_code=station,
                                           location_code=location,
                                           channel_code=channel)

            out_picks.append(
                Pick(waveform_id=waveform_id, phase_hint=pick.phase,
                     time=UTCDateTime(pick.datetime),
                     evaluation_mode='automatic',
                     evaluation_status='preliminary',
                     method_id=ResourceIdentifier('beamforming'),
                     method='beamforming'))

        return out_picks

    def add_picker_model_from_file(self, file_path):
        # shutil.copyfile(file_path, self.files.picker_model)
        self.ai_picker = AIPicker.read(file_path)
        self.ai_picker.write(self.files.picker_model)

    def add_picker_model(self, model):
        self.ai_picker = model
        self.ai_picker.write(self.files.picker_model)

    def synthetic_pick(self, origin_location, origin_time):
        tts = self.travel_times.travel_time(origin_location)

        picks = []
        for phase in ['P', 'S']:
            for tt in tts[phase]:
                t = origin_time + tt
                waveform_id = WaveformStreamID(network_code=self.network_code,
                                               station_code=self.station_code)
                # pk = Pick(time=t, )
                picks.append()



    def snr_repick(self, stream: uquake.core.stream.Stream,
                   picks: List[uquake.core.event.Pick],
                   ai_enhanced=False,
                   setting_section='snr_repicker'):

        """
        Repick using a snr based ensemble picker
        :param stream: stream containing the waveforms
        :type stream: :class: uquake.core.stream.Stream
        :param picks: a list of picks associated to the waveforms
        :type picks: a list of :class: uquake.core.event.Pick
        :return:
        """

        start_search_window = self.settings[
            setting_section].start_search_window
        end_search_window = self.settings[setting_section].end_search_window
        start_refined_search_window = \
            self.settings[setting_section].start_refined_search_window
        end_refined_search_window = \
            self.settings[setting_section].end_refined_search_window
        search_resolution = self.settings[setting_section].search_resolution
        pre_pick_window_len = self.settings[
            setting_section].pre_pick_window_len
        post_pick_window_len = self.settings[
            setting_section].post_pick_window_len

        snrs, new_picks = \
            snr_ensemble_re_picker(stream, picks,
                                   start_search_window=start_search_window,
                                   end_search_window=end_search_window,
                                   start_refined_search_window=
                                   start_refined_search_window,
                                   end_refined_search_window=
                                   end_refined_search_window,
                                   refined_window_search_resolution=
                                   search_resolution,
                                   snr_calc_pre_pick_window_len
                                   =pre_pick_window_len,
                                   snr_calc_post_pick_window_len
                                   =post_pick_window_len)

        out_picks = []
        for _i, new_pick in enumerate(new_picks):
            network = new_pick.waveform_id.network_code
            station = new_pick.waveform_id.station_code
            location = new_pick.waveform_id.location_code
            channel = new_pick.waveform_id.channel_code

            pre_wl = self.settings.snr_repicker.snr_pre_pick_window
            post_wl = self.settings.snr_repicker.snr_post_pick_window

            snr = calculate_snr(stream.select(network=network, station=station,
                                              location=location)[0],
                                new_pick.time, pre_wl=pre_wl, post_wl=post_wl)

            # if snr < self.settings.snr_repicker.snr_threshold:
            #     continue

            new_pick.snr = snr
            out_picks.append(new_pick)
            
        if ai_enhanced:
            if self.ai_picker is None:
                logger.error('no picker picker model in the project\n'
                             'to add a picker picker model please use the'
                             'add_ai_picker_from_file method')
            else:
                self.ai_pick(stream.copy(), out_picks)

        return PickerResult(new_picks=out_picks,
                            initial_picks=picks,
                            snr_threshold=
                            self.settings.snr_repicker.snr_threshold,
                            stream=stream)

    @staticmethod
    def origin_time_correction_from_picks(stream: Stream, picks: List[Pick],
                                          search_window: float = 0.2,
                                          kurtosis_window: float = 0.02,
                                          use_p: bool = True,
                                          use_s: bool = True,
                                          snr_threshold: float = None)\
            -> PickerResult:
        """
        estimate the origin time using the waveform and a list of picks.
        This function is used to correct the origin time obtained from
        beam forming.
        :param stream: the waveforms
        :type stream: uquake.core.stream.Stream
        :param picks: a list of picks
        :type picks: List[Pick]
        :param search_window: search window in seconds
        :type search_window: float
        :param kurtosis_window: length of the window used to calculate the
        Kurtosis function. see obspy.realtime.signal import kurtosis
        :type kurtosis_window: float
        :param use_p: use p picks
        :type use_p: bool
        :param use_s: use s picks
        :type use_s: bool
        :param snr_threshold: snr threshold for picks
        :type snr_threshold: float
        :return: time correction in second
        :rtype: uquake.core.UTCDateTime
        """

        wf = stream
        min_time = np.min([tr.stats.starttime for tr in wf])

        sampling_rate = wf[0].stats.sampling_rate
        window_length = int(search_window * sampling_rate)

        # stack data
        stacked_data = np.zeros(len(wf[0].data))
        for pick in picks:
            if (pick.phase_hint == 'P') and not use_p:
                continue
            if (pick.phase_hint == 'S') and not use_s:
                continue
            station = pick.waveform_id.station_code
            wf = wf.resample(sampling_rate)
            for tr in wf.select(station=station):
                if (tr.stats.starttime > pick.time) or \
                        (pick.time > tr.stats.endtime):
                    continue
                data = tr.data.astype(np.float32) ** 2
                data /= np.std(data)
                n_sample = int((min_time - pick.time) * tr.stats.sampling_rate)
                stacked_data += fftshift(np.roll(data, n_sample))

        stacked_trace = tr.copy()
        stacked_trace.data = stacked_data

        i_max = np.argmax(stacked_trace)

        k = kurtosis(stacked_trace, win=kurtosis_window)
        diff_k = np.diff(k)
        i_0 = int(i_max - window_length / 2)
        if i_0 < 0:
            i_0 = 0
        i_1 = int(i_max + window_length / 2)
        if i_1 > len(stacked_trace):
            i_1 = len(stacked_trace)
        o_i = np.argmax(np.abs(diff_k[i_0: i_1])) + i_0
        origin_time_correction = (o_i - len(stacked_trace) / 2) / sampling_rate

        out_picks = []
        for pick in picks:
            pick.time += origin_time_correction
            out_picks.append(pick)

        return PickerResult(out_picks, picks, snr_threshold, stream)

    def ai_pick(self, st: Stream, picks: list):
        import matplotlib.pyplot as plt

        st2 = st.copy()
        st2 = st2.detrend('demean').detrend('linear').taper(max_length=0.01,
                                                            max_percentage=1)

        st2 = st2.resample(sampling_rate=self.settings.ai_picker.sampling_rate)

        picks_out = []
        for pick in picks:
            out_pick = pick.copy()
            network = pick.waveform_id.network_code
            station = pick.waveform_id.station_code
            location = pick.waveform_id.location_code
            phase = pick.phase_hint

            pick_time = pick.time

            st3 = st2.copy().select(network=network, station=station,
                                    location=location)

            tr = st2.copy().select(network=network, station=station,
                                   location=location).composite()[0]

            predicted_time = self.ai_picker.predict_trace(tr.copy(),
                                                          pick_time,
                                                          phase)

            pick.time = predicted_time
            picks_out.append(pick)

        return picks_out


def measure_linearity_planarity(st: Stream, pick: Pick, window_length: float):
    network = pick.waveform_id.network_code
    station = pick.waveform_id.station_code
    location = pick.waveform_id.location_code

    st2 = st.select(network=network, station=station, location=location)

    wave_mat = []
    for tr in st2:
        window_start_sample = int((pick.time - tr.stats.starttime)
                                  * tr.stats.sampling_rate)
        window_end_sample = int(window_start_sample +
                                window_length * tr.stats.sampling_rate)

        wave_mat.append(tr.data[window_start_sample: window_end_sample])

    cov_mat = np.cov(np.array(wave_mat))

    eig_vals, eig_vects = np.linalg.eig(cov_mat)
    vals = np.sort(eig_vals)

    if pick.phase_hint == 'P':
        out_value = 1 - np.linalg.norm([vals[0], vals[1]]) / vals[2]
    else:
        out_value = 1 - vals[0] / np.linalg.norm([vals[1], vals[2]])
    return out_value



# def measure_planarity(st: Stream, pick: Pick, window_length):
#     st.rotate('->')
#     pass







