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
from pydantic import BaseModel, validator, ValidationError
from typing import Optional, List


class ResultRepickerSNR(object):
    def __init__(self, new_picks, original_picks):
        self.new_picks = new_picks
        self.original_picks = original_picks

    def append_events(self, event: uquake.core.event.Event = None,
                      snr_threshold: float=None):

        arrivals = []
        for pick in self.new_picks:

            arrival = Arrival(pick_id=pick.resource_id,
                              phase=pick.phase_hint)
            arrivals.append(arrival)

            origin = Origin(arrivals=arrivals)

        if event is None:
            event = Event(picks=self.picks, origin=origin)

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


class RepickerSNR(ProjectManager):

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

        self.files.snr_repicker_settings = self.paths.config / \
                                        'snr_repicker_settings.toml'

        if not self.files.snr_repicker_settings.is_file():
            settings_template = Path(os.path.realpath(__file__)).parent / \
                            '../settings/snr_repicker_settings_template.toml'

            shutil.copyfile(settings_template,
                            self.files.snr_repicker_settings)

        self.settings = Settings(settings_location=self.paths.config,
                                 settings_file=
                                 self.files.snr_repicker_settings.name)

    @staticmethod
    def convert_nmx_picks(nmx_picks):
        out_picks = []
        for pick in nmx_picks.iterrows():
            pick = pick[1]
            network, station, _ = pick.site_id.split('.')
            location = '00'

            waveform_id = WaveformStreamID(network_code=network,
                                           station_code=station,
                                           location_code=location)

            out_picks.append(
                Pick(waveform_id=waveform_id, phase_hint=pick.phase,
                     time=UTCDateTime(pick.datetime),
                     evaluation_mode='automatic',
                     evaluation_status='preliminary',
                     method_id=ResourceIdentifier('beamforming'),
                     method='beamforming'))

        return out_picks

    def re_pick(self, stream: uquake.core.stream.Stream,
                picks: uquake.core.event.Pick):

        """
        Repick using a snr based ensemble picker
        :param stream: stream containing the waveforms
        :type stream: :class: uquake.core.stream.Stream
        :param picks: a list of picks associated to the waveforms
        :type picks: a list of :class: uquake.core.event.Pick
        :return:
        """

        start_search_window = self.settings.snr_repicker.start_search_window
        end_search_window = self.settings.snr_repicker.end_search_window
        start_refined_search_window = \
            self.settings.snr_repicker.start_refined_search_window
        end_refined_search_window = \
            self.settings.snr_repicker.end_refined_search_window
        search_resolution = self.settings.snr_repicker.search_resolution
        pre_pick_window_len = self.settings.snr_repicker.pre_pick_window_len
        post_pick_window_len = self.settings.snr_repicker.post_pick_window_len

        snrs, new_picks = snr_ensemble_re_picker(stream, picks,
                                                 start_search_window,
                                                 end_search_window,
                                                 start_refined_search_window,
                                                 end_refined_search_window,
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

            pre_wl = self.settings.snr_repicker.snr_pre_pick_window
            post_wl = self.settings.snr_repicker.snr_post_pick_window

            snr = calculate_snr(stream.select(network=network, station=station,
                                              location=location)[0],
                                new_pick.time, pre_wl=pre_wl, post_wl=post_wl)

            if snr < self.settings.snr_repicker.snr_threshold:
                continue

            new_pick.snr = snr
            out_picks.append(new_pick)

        return ResultRepickerSNR(new_picks=out_picks,
                                 origin_picks=picks)
