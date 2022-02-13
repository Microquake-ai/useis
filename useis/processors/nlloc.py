import openpyxl.styles.stylesheet
import uquake.core.inventory
from uquake.core.inventory import Inventory

from ..core.project_manager import *
from uquake.nlloc.nlloc import *
from uquake.core.event import (Catalog, Event, CreationInfo, Origin, Arrival,
                               Pick, WaveformStreamID)
from uquake.core import UTCDateTime
from uquake.core.stream import Stream
import numpy as np
import sys
import toml
# from pydantic import BaseModel
# from ..services import models
# from typing import Optional, List
import matplotlib.pyplot as plt


def locate_hodogram(st: Stream, event: Event, inventory: Inventory,
                    window_length: float = 20e-3, linearity_threshold=0.7):
    """
    measure the incidence angle from hodogram
    :param st: a stream object containing the waveforms
    :type st: uquake.core.stream.Stream
    :param picks: a list of picks associated the the waveforms
    :type picks: list of uquake.core.event.Picks
    :param inventory: inventory file
    :type inventory: uquake.core.inventory.Inventory
    :param window_length: lenght of the window within which the hodogram is
    measured
    :return:
    """

    # Rotate the stream in "E", "N", "Z" if that is not yet the case
    st_zne = st.copy().rotate('->ZNE', inventory=inventory)
    st_zne = st_zne.filter('highpass', freq=10)

    # for each pick measure the incidence angle using the hodogram
    if event.preferred_origin() is not None:
        origin = event.preferred_origin()
    else:
        origin = event.origins[-1]

    azimuths = []
    plunges = []
    linearities = []
    phases = []
    scatters = event.preferred_origin().scatter
    dot_prod = np.zeros(len(scatters))
    for arrival in origin.arrivals:
        pick = arrival.pick
        network = pick.waveform_id.network_code
        station = pick.waveform_id.station_code
        location = pick.waveform_id.location_code

        start_time = pick.time
        end_time = pick.time + window_length

        st_tmp = st_zne.copy().select(network=network, station=station,
                                      location=location).trim(
            starttime=start_time, endtime=end_time)

        wave_mat = []
        for component in ['E', 'N', 'Z']:
            tr = st_tmp.select(component=component)[0]
            wave_mat.append(tr.data)

        wave_mat = np.array(wave_mat)
        # plt.figure(2)
        # plt.close('all')
        # plt.clf()
        # plt.plot(wave_mat[1, :] / np.max(wave_mat[1, :]),
        #          wave_mat[0, :] / np.max(wave_mat[0, :]), 'k')

        cov_mat = np.cov(np.array(wave_mat))

        e = None

        eig_vals, eig_vects = np.linalg.eig(cov_mat)
        i_ = np.argsort(eig_vals)

        if arrival.phase == 'P':
            eig_vect = eig_vects[i_[-1]]
            linearity =  (1 - np.linalg.norm(eig_vals[i_[:2]]) /
                          eig_vals[i_[2]])
            # input(f'grossomodo P {linearity}')
            color = 'b'
        elif arrival.phase == 'S':
            eig_vect = eig_vects[i_[0]]
            linearity = (1 - eig_vals[i_[0]] /
                        np.linalg.norm(eig_vals[i_[1:]]))
            color = 'r'

        # if linearity < linearity_threshold:
        #     continue

        sta = inventory.select(network=network,
                               station=station,
                               location=location)[0][0]

        eig_vect /= np.linalg.norm(eig_vect)

        for i, scat in enumerate(scatters):
            vect = sta.loc - scat[0:3]
            vect /= np.linalg.norm(vect)
            dot_prod[i] += np.abs(np.dot(vect, eig_vect)) * scat[-1] * \
                           linearity

    return scatters[np.argmax(dot_prod), :-1]


def calculate_uncertainty(point_cloud):
    """
    :param point_cloud: location point cloud
    :type point_cloud: numpy 2d array
    :return: uncertainty
    :rtype: uquake.core.event.OriginUncertainty
    """

    # horizontal uncertainty
    if len(point_cloud) == 0:
        ce = ConfidenceEllipsoid(semi_major_axis_length=0,
                                 semi_intermediate_axis_length=0,
                                 semi_minor_axis_length=0,
                                 major_axis_plunge=0,
                                 major_axis_azimuth=0,
                                 major_axis_rotation=0)

        return OriginUncertainty(horizontal_uncertainty=0,
                                 confidence_ellipsoid=0,
                                 preferred_description='confidence ellipsoid',
                                 confidence_level=68)

    v, u = np.linalg.eig(np.cov(point_cloud[:, 0:2].T))
    horizontal_uncertainty = np.sqrt(np.max(v))

    v, u = np.linalg.eig(np.cov(point_cloud.T))

    major_axis_index = np.argmax(v)

    uncertainty = np.sort(np.sqrt(v))[-1::-1]

    h = np.linalg.norm(u[major_axis_index, :-1])
    vert = u[major_axis_index, -1]

    major_axis_plunge = np.arctan2(-vert, h)
    x = u[major_axis_index, 0]
    y = u[major_axis_index, 1]
    major_axis_azimuth = np.arctan2(x, y)
    major_axis_rotation = 0

    ce = ConfidenceEllipsoid(semi_major_axis_length=uncertainty[0],
                             semi_intermediate_axis_length=uncertainty[1],
                             semi_minor_axis_length=uncertainty[2],
                             major_axis_plunge=major_axis_plunge,
                             major_axis_azimuth=major_axis_azimuth,
                             major_axis_rotation=major_axis_rotation)

    return OriginUncertainty(horizontal_uncertainty=horizontal_uncertainty,
                             confidence_ellipsoid=ce,
                             preferred_description='confidence ellipsoid',
                             confidence_level=68)


class NLLOCResult(object):

    def __init__(self, nll_object, hypocenter: np.array,
                 event_time: UTCDateTime, scatter_cloud: np.ndarray,
                 rays: list, observations: Observations,
                 evaluation_mode: str, evaluation_status: str,
                 hypocenter_file: str):
        self.nll_object = nll_object
        self.hypocenter = hypocenter
        self.hypocenter_global = (self.nll_object.projection.
                                  transform_to_global(hypocenter[0],
                                                      hypocenter[1],
                                                      hypocenter[2]))
        self.event_time = event_time
        self.scatter_cloud = scatter_cloud
        self.rays = rays
        self.observations = observations
        self.evaluation_mode = evaluation_mode
        self.evaluation_status = evaluation_status

        self.origin_uncertainty = calculate_uncertainty(
            self.scatter_cloud[:, :-1])

        self.creation_info = CreationInfo(author='uQuake-nlloc',
                                          creation_time=UTCDateTime.now())
        self.hypocenter_file = hypocenter_file
        self.unc = self.origin_uncertainty.confidence_ellipsoid.\
            semi_major_axis_length

    def __repr__(self):
        out_str = f"""
        time (UTC)  : {self.t}
        location    : x- {self.x:>10.1f} (m)
                      y- {self.y:>10.1f} (m)
                      z- {self.z:>10.1f} (m)
        uncertainty : {self.uncertainty:0.1f} (1 std - m)
        """
        return out_str

    def __add__(self, other: Event):
        if not (isinstance(other, Catalog) | isinstance(other, Event)):
            raise TypeError(f'object type {type(Event)} or'
                            f'{type(Catalog)} expected. Object of type '
                            f'{type(other)} provided')
        return self.append_to_event(other)

    @property
    def loc(self):
        return self.hypocenter

    @property
    def latitude(self):
        return self.hypocenter_global[1]

    @property
    def longitude(self):
        return self.hypocenter_global[0]

    @property
    def depth(self):
        return self.hypocenter_global[2]

    @property
    def x(self):
        return self.loc[0]

    @property
    def y(self):
        return self.loc[1]

    @property
    def z(self):
        return self.loc[2]

    @property
    def t(self):
        return self.event_time

    @property
    def time(self):
        return self.event_time

    @property
    def arrivals(self) -> list:
        arrivals = []
        predicted_times = self.nll_object.travel_times.travel_time(
            self.hypocenter)
        angles = self.nll_object.travel_times.angles(self.hypocenter)
        for pick in self.observations.picks:
            travel_time = pick.time - self.t
            phase = pick.phase_hint
            site_code = pick.site

            network = pick.waveform_id.network_code
            station = pick.waveform_id.station_code
            location = pick.waveform_id.location_code

            if self.rays is not None:
                for ray in self.rays:
                    if (ray.site_code == site_code) & (ray.phase == phase):
                        break
                distance = ray.length

                time_residual = Arrival.calculate_time_residual(
                    ray.travel_time,
                    travel_time)

                azimuth = ray.azimuth
                takeoff_angle = ray.takeoff_angle

            else:
                inv = self.nll_object.inventory.select(network=network,
                                                       station=station,
                                                       location=location)

                distance = np.linalg.norm(self.hypocenter - inv[0][0][0].loc)

                predicted_time = predicted_times[phase][site_code]

                time_residual = Arrival.calculate_time_residual(predicted_time,
                                                                travel_time)

                azimuth = angles['azimuth'][phase][site_code]
                takeoff_angle = angles['takeoff'][phase][site_code]

            time_weight = 1

            arrival = Arrival(phase=phase, distance=distance,
                              time_residual=time_residual,
                              time_weight=time_weight, azimuth=azimuth,
                              takeoff_angle=takeoff_angle,
                              pick_id=pick.resource_id)

            arrivals.append(arrival)

        return arrivals

    @property
    def origin(self):
        origin = Origin(x=self.x, y=self.y, z=self.z, time=self.t,
                        latitude=self.latitude,
                        longitude=self.longitude,
                        depth=self.depth,
                        evaluation_mode=self.evaluation_mode,
                        evaluation_status=self.evaluation_status,
                        epicenter_fixed=False, method_id='uQuake-NLLOC',
                        creation_info=self.creation_info,
                        arrivals=self.arrivals,
                        origin_uncertainty=self.origin_uncertainty,
                        uncertainty=self.unc,
                        rays=self.rays,
                        scatter=self.scatter_cloud)
        origin.scatter = self.scatter_cloud
        origin.rays = self.rays
        return origin

    @property
    def event(self):
        return self.export_as_event()

    @property
    def uncertainty(self):
        ce = self.origin_uncertainty.confidence_ellipsoid
        return ce.semi_major_axis_length

    @property
    def predicted_picks(self):
        if not self.rays:
            return

        predicted_picks = []
        for ray in self.rays:
            predicted_picks.append(ray.to_pick(self.time))

        return predicted_picks

    def append_to_event(self, event: Event) -> Event:
        o = self.origin
        if isinstance(event, Catalog):
            event[0].append_origin_as_preferred_origin(o)
        elif isinstance(event, Event):
            event.append_origin_as_preferred_origin(o)
        return event

    def export_as_event(self):
        o = self.origin
        e = Event(origins=[o], picks=self.observations.picks)
        e.preferred_origin_id = o.resource_id
        return e

    def to_2d(self, station):
        return NLLOCResult2DCylindrical.from_nlloc_result(self, station)

    def relocate(self, residual_threshold=10e-3):
        event = self.event
        picks = []
        for arrival in event.preferred_origin().arrivals:
            if arrival.time_residual < residual_threshold:
                picks.append(arrival.pick)

        observations = Observations(picks=picks)
        return self.nll_object.run_location(observations)

    def relocate_hodogram(self, stream: Stream, window_length: float = 20e-3):
        loc = locate_hodogram(stream, self.event, self.nll_object.inventory,
                               window_length=window_length)

        self.hypocenter = np.array(loc)

        return loc


class NLLOCResult2DCylindrical(NLLOCResult):
    """
    Transform a NLLOCResult object into 2D Cylindrical solution with more
    appropriate measurement of the uncertainty.
    This object is used mainly in the case of a single borehole array.
    It is assumed that the borehole is nearly vertical and that the deviation
    can be neglected
    """

    def __init__(self, nlloc_object, hypocenter: np.array,
                 event_time: UTCDateTime,
                 scatter_cloud: np.ndarray, rays: list,
                 observations: Observations, evaluation_mode: str,
                 evaluation_status: str, hypocenter_file: str,
                 inventory: uquake.core.inventory.Inventory, station: str):

        """
        :param nlloc_object:
        :param hypocenter: event hypocenter location
        :param event_time: event time
        :param scatter_cloud: cloud of probable location
        :param rays: a list of ray
        :param observations: travel time observations used to calculate the
        location
        :param evaluation_mode: evaluation mode as defined by the QuakeML
        standard
        :param evaluation_status: evaluation status as defined by the QuakeML
        standard
        :param hypocenter_file: The hypocenter files output by NLLoc
        :param inventory: an inventory file
        :param station: station name
        """

        inv = inventory.select(station=station)
        xs = []
        ys = []
        for site in inv.sites:
            xs.append(site.x)
            ys.append(site.y)

        self.reference_x = np.mean(xs)
        self.reference_y = np.mean(ys)

        super().__init__(nlloc_object, hypocenter, event_time, scatter_cloud,
                        rays, observations, evaluation_mode, evaluation_status,
                         hypocenter_file)

    @classmethod
    def from_nlloc_result(cls, nlloc_result: NLLOCResult, station: str):
        return cls(nlloc_result.nll_object,nlloc_result.hypocenter,
                   nlloc_result.event_time,
                   nlloc_result.scatter_cloud, nlloc_result.rays,
                   nlloc_result.observations, nlloc_result.evaluation_mode,
                   nlloc_result.evaluation_status,
                   nlloc_result.hypocenter_file,
                   nlloc_result.nll_object.inventory, station)

    @property
    def uncertainty_h(self):
        xs = self.scatter_cloud[:, 0] - self.reference_x
        ys = self.scatter_cloud[:, 1] - self.reference_y
        return np.std(np.sqrt(xs ** 2 + ys ** 2))

    @property
    def uncertainty_z(self):
        return np.std(self.scatter_cloud[:, -2])

    @property
    def uncertainty(self):
        return np.sqrt(self.uncertainty_h ** 2 + self.uncertainty_z ** 2)

    @property
    def r(self):
        return np.linalg.norm([self.x - self.reference_x,
                               self.y - self.reference_y])

    @property
    def origin(self):

        self.origin_uncertainty.horizontal_uncertainty = self.uncertainty_h

        origin = Origin(x=self.r, y=0, z=self.z, time=self.t,
                        evaluation_mode=self.evaluation_mode,
                        evaluation_status=self.evaluation_status,
                        epicenter_fixed=False, method_id='uQuake-NLLOC',
                        creation_info=self.creation_info,
                        arrivals=self.arrivals,
                        origin_uncertainty=self.origin_uncertainty,
                        uncertainty=self.uncertainty_h,
                        rays=self.rays,
                        scatter=self.scatter_cloud)
        origin.scatter = self.scatter_cloud
        origin.rays = self.rays
        return origin

    def __repr__(self):
        out_str = f"""
        time (UTC)             : {self.t}
        location               : r- {self.r:>10.1f} (m)
                                 z- {self.z:>10.1f} (m)
        uncertainty Horizontal : {self.uncertainty_h:0.1f} (1 std - m)
        uncertainty Vertical   : {self.uncertainty_z:0.1f} (1 std - m)
        uncertainty            : {self.uncertainty:0.1f} (1 std - m)
        """
        return out_str

    def to_3d(self, azimuth: float, std_azimuth: float) -> NLLOCResult:
        """
        :centroi
        :param azimuth: azimuth in degrees
        :param std_azimuth: standard deviation of the Azimuth in degrees
        :return:
        """

        # converting degrees to radians

        azimuth = (azimuth * np.pi / 180) % (2 * np.pi)

        x = np.sin(azimuth) * self.r + self.reference_x  # Northing
        y = np.cos(azimuth) * self.r + self.reference_y  # Easting

        hypocenter = np.array([x, y, self.z])

        d_s_x = self.scatter_cloud[:, 0] - self.reference_x
        d_s_y = self.scatter_cloud[:, 1] - self.reference_y

        scatter_azimuths = (np.arctan2(d_s_x, d_s_y)) % (2 * np.pi)

        min_azimuth = (azimuth - std_azimuth) % (2 * np.pi)
        max_azimuth = (azimuth + std_azimuth) % (2 * np.pi)

        scatter_cloud = self.scatter_cloud[
            (scatter_azimuths >= min_azimuth) &
            (scatter_azimuths <= max_azimuth), :]

        if len(scatter_cloud) == 0:
            min_azimuth = (azimuth - 10 / 180 * np.pi) % (2 * np.pi)
            max_azimuth = (azimuth + 10 / 180 * np.pi) % (2 * np.pi)

            scatter_cloud = self.scatter_cloud[
                            (scatter_azimuths >= min_azimuth) &
                            (scatter_azimuths <= max_azimuth), :]

        return NLLOCResult(self.nll_object, hypocenter, self.time,
                           scatter_cloud, self.rays, self.observations,
                           self.evaluation_mode, self.evaluation_status,
                           self.hypocenter_file)


class NLLOC(ProjectManager):
    def __init__(self, base_projects_path: Path, project_name: str,
                 network_code: str, use_srces: bool=False):

        """
        Object to control NLLoc execution and manage the required grids, inputs
        and outputs
        :param base_projects_path:
        :param project_name:
        :param network_code:
        :param use_srces:
        """

        super().__init__(base_projects_path, project_name, network_code,
                         use_srces=use_srces)

        self.run_id = str(uuid4())

        self.paths.current_run = self.paths.root / 'run' / self.run_id
        self.paths.current_run.mkdir(parents=True, exist_ok=False)

        self.paths.outputs = self.paths.current_run / 'outputs'
        self.paths.outputs.mkdir(parents=True, exist_ok=True)

        self.paths.observations = self.paths.current_run / 'inputs'
        self.paths.observations.mkdir(parents=True, exist_ok=True)
        self.files.observations = self.paths.observations / 'picks.obs'
        self.observations = None

        self.paths.templates = self.paths.root / 'templates'
        self.paths.templates.mkdir(parents=True, exist_ok=True)

        self.files.control = self.paths.current_run / 'run.nll'

        self.last_event_hypocenter = None
        self.last_event_time = None

        self.files.nlloc_settings = self.paths.config / 'nlloc_settings.py'

        sys.path.append(str(self.paths.config))

        if not self.files.nlloc_settings.is_file():
            settings_template = Path(os.path.realpath(__file__)).parent / \
                                    '../settings/nlloc_settings_template.py'

            shutil.copyfile(settings_template, self.files.nlloc_settings)

        self.nlloc_settings = __import__('nlloc_settings')
        self.control_template = self.nlloc_settings.nlloc_control

        self.settings = Settings(str(self.paths.config))

    def __del__(self):
        self.remove_run_directory()

    def remove_run_directory(self):
        for fle in self.paths.observations.glob('*'):
            fle.unlink()

        if self.paths.observations.exists():
            self.paths.observations.rmdir()

        output_dir = self.paths.outputs

        for fle in output_dir.glob('*'):
            fle.unlink()

        if output_dir.exists():
            output_dir.rmdir()

        for fle in self.paths.outputs.parent.glob('*'):
            fle.unlink()

        if self.paths.outputs.parent.exists():
            self.paths.outputs.parent.rmdir()
        else:
            logger.info(f'{self.paths.outputs.parent} was already deleted...'
                        f'nothing to do')

    def write_control_file(self):
        with open(self.files.control, 'w') as control_file:
            control_file.write(self.control)

    def __add_observations__(self, observations):
        """
        adding observations to the project
        :param observations: Observations
        """
        if not isinstance(observations, Observations):
            raise TypeError(f'observations is type {type(observations)}. '
                            f'observations must be type {Observations}.')
        self.paths.observations.mkdir(parents=True, exist_ok=True)
        observations.write(self.files.observations.name,
                           path=self.paths.observations)

        self.observations = observations

    def run_location(self, observations=None, calculate_rays=True,
                     delete_output_files=True, event=None,
                     evaluation_mode: str = 'automatic',
                     evaluation_status: str = 'preliminary',
                     multithreading=False):

        import subprocess

        if event is not None:
            observations = Observations.from_event(event=event)

        if (observations is None) and (self.observations is None):
            raise ValueError('The current run does not contain travel time'
                             'observations. Observations should be added to '
                             'the current run using the add_observations '
                             'method.')

        elif observations is not None:
            self.__add_observations__(observations)

        self.write_control_file()

        cmd = ['NLLoc', self.files.control]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)

        logger.info('locating event using NonLinLoc')
        t0 = time()
        output, error = process.communicate()
        if error:
            raise Exception(error.decode('ascii'))
        logger.info(output.decode('ascii'))
        t1 = time()
        logger.info(f'done locating event in {t1 - t0:0.2f} seconds')

        if event is not None:
            if isinstance(event, Catalog):
                event = event[0]
            if event.preferred_origin() is not None:
                event_time = event.preferred_origin().time
            else:
                event_time = event.origins[-1].time
        else:
            p_times = []
            for pick in observations.picks:
                p_times.append(pick.time)
            if not p_times:
                logger.error('empty results')
                return
            event_time = np.min(p_times)

        if not (self.paths.outputs / 'last.hyp').exists():
            logger.error(f'event location failed for event {event_time}!')
            return

        with open(self.paths.outputs / 'last.hyp') as hyp_file:
            hypocenter_file = hyp_file.readlines()

        t, x, y, z = read_hypocenter_file(self.paths.outputs / 'last.hyp')

        eloc = np.array([x, y, z])

        scatters = read_scatter_file(self.paths.outputs / 'last.scat')

        rays = None
        if calculate_rays:
            logger.info('calculating rays')
            t0_ray = time()
            rays = self.travel_times.ray_tracer(np.array([x, y, z]),
                                                multithreading=multithreading)
            t1_ray = time()
            logger.info(f'done calculating rays in {t1_ray - t0_ray:0.2f} '
                        f'seconds')

        # for travel_time in self.travel_times.travel_time(eloc, )

        if delete_output_files:
            self.remove_run_directory()

        result = NLLOCResult(self, eloc, t, scatters, rays, observations,
                             evaluation_mode, evaluation_status,
                             hypocenter_file)

        return result

    def rays(self, hypocenter_location):
        return self.travel_times.ray_tracer(hypocenter_location)

    @property
    def nlloc_files(self):
        return NllocInputFiles(self.files.observations,
                               self.paths.times /
                               self.network_code,
                               self.paths.outputs / self.network_code)

    @property
    def control(self):

        if self.srces is None:
            raise ValueError('The project does not contain sites or '
                             'inventory. sites (srces) or inventory '
                             'information can be added using the add_srces or'
                             'add_inventory methods.')

        if self.observations is None:
            raise ValueError('The current run does not contain travel time '
                             'observations. Observations should be added to '
                             'the current run using the add_observations '
                             'method.')

        observations = str(self.observations)

        ctrl = ''
        ctrl += str(self.control_template['control']) + '\n'
        ctrl += str(self.control_template['transformation']) + '\n\n'

        if self.control_template['locsig'] is not None:
            ctrl += self.control_template['locsig']

        if self.control_template['loccom'] is not None:
            ctrl += self.control_template['loccom'] + '\n'

        ctrl += str(self.srces) + '\n'

        ctrl += str(self.nlloc_files) + '\n'

        ctrl += str(self.control_template['locsearch'])
        ctrl += str(self.control_template['locmeth'])
        ctrl += str(self.control_template['locgau']) + '\n'

        ctrl += str(self.control_template['locqual2err'])

        if self.p_velocity is not None:
            ctrl += str(LocGrid.init_from_grid(self.p_velocity))
        else:
            raise ValueError('Cannot initialize the LocGrid, the velocity '
                             'grids are not defined')

        return ctrl
