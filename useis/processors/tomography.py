import pandas as pd

from ..core.project_manager import ProjectManager
from uquake.grid.nlloc import VelocityGrid3D, VelocityGridEnsemble
from uquake.nlloc.nlloc import Srces, Site, Observations
from uquake.core.event import (Catalog, Origin, Arrival, Pick,
                               WaveformStreamID, Ray, ResourceIdentifier)
from uquake.core import UTCDateTime
from uquake.core.logging import logger
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from pydantic import BaseModel, conlist
from enum import Enum
from pydantic.typing import List, Union
from datetime import datetime, timedelta
from uuid import uuid4
from functools import partial
import scipy as sc
from scipy.sparse import csr_matrix
import docker
from ..tomography import data as estuaire_data
import os
import pickle
from pathlib import Path

__cpu_count__ = cpu_count()


class Phase(str, Enum):
    p = 'P'
    s = 'S'

    def __str__(self):
        return self.value

    def __expr__(self):
        return self.__str__()

    def __call__(self):
        return self.value


class SyntheticTypes(str, Enum):
    random = 'random'
    cubic = 'cubic'

    def __str__(self):
        return self.value

    def __expr__(self):
        return self.__str__()

    def __call__(self):
        return self.value


class EventData(BaseModel):
    location: conlist(float, min_items=3, max_items=3)
    location_correction: conlist(float, min_items=3, max_items=3) = [0, 0, 0]
    origin_time: datetime
    resource_id: str = None
    origin_time_correction: float = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resource_id = ResourceIdentifier()

    @property
    def loc(self):
        return np.array(self.location)

    @property
    def id(self):
        return self.resource_id.id

    @property
    def time_correction(self):
        return timedelta(self.origin_time_correction)

    @property
    def time(self):
        return self.origin_time + self.time_correction

    def __str__(self):
        return f'{self.id},{self.location},{self.location_correction}'

    def __repr__(self):
        return str(self)
    

class EventEnsemble(object):
    def __init__(self, events: List[EventData] = []):
        self.events = events
        self.dict = {}
        self.__ids__ = list(np.arange(len(events) + 1))
        for event in events:
            self.dict[event.resource_id] = event

    def __len__(self):
        return len(self.events)
    
    def __getitem__(self, items):
        if isinstance(items, (list, np.ndarray)):
            events = []
            for item in items:
                for event in self.events:
                    if isinstance(item, str):
                        if event.resource_id.id == item:
                            events.append(event)
                    elif isinstance(item, int):
                        events.append(self.events[item])
                    elif isinstance(item, ResourceIdentifier):
                        if event.resource_id == item:
                            events.append(event)
                    else:
                        raise ValueError('the indices must be str, int or'
                                         'ResourceIdentifier')
            return EventEnsemble(events=events)
        elif isinstance(items, int):
            return self.events[items]
        elif isinstance(items, ResourceIdentifier):
            for event in self.events:
                if event.resource_id == items:
                    return event
        elif isinstance(items, str):
            for event in self.events:
                if event.resource_id.id == items:
                    return event
        else:
            raise TypeError('the index must be a str, an int or a '
                            'ResourceIdentifier')

    def __str__(self):
        out_str = ''
        for event in self.events:
            out_str += f'{event}\n'
        return out_str

    def __repr__(self):
        return str(self)

    def get_index(self, item):
        if isinstance(item, ResourceIdentifier):
            for _id, event in zip(self.__ids__, self.events):
                if event.resource_id == item:
                    return _id
        elif isinstance(item, str):
            for _id, event in zip(self.__ids__, self.events):
                if event.resource_id.id == item:
                    return _id

    def select(self, ids: Union[List[int], int]):
        return self[ids]
    
    def append(self, event: EventData):
        self.__ids__.append(len(self) + 1)
        self.events.append(event)
        self.dict[event.resource_id] = event

    @property
    def locs(self):
        locs = []
        for event in self.events:
            locs.append(event.location)
        return locs


class ArrivalTimeData(BaseModel):
    event_id: int
    site_name: str
    site_id: int
    phase: Phase
    resource_id: str = None
    arrival_time: datetime

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resource_id = ResourceIdentifier()

    @property
    def id(self):
        return self.resource_id.id

    def __add__(self, other):

        if not isinstance(other, type(self)):
            raise TypeError(f'operation not permitted between {type(self)} '
                            f'and {type(other)}')

        if self.event_id != other.event_id:
            raise ValueError(f'operation only permitted between two objects'
                             f'with the same event id')

        return self.arrival_time + other.arrival_time

    def __repr__(self):
        return f'event id: {self.event_id}, site id: {self.site_id}, ' \
               f'arrival time:{self.arrival_time}'


class ArrivalTimeEnsemble(object):
    def __init__(self, arrival_times:
    Union[List[ArrivalTimeData], np.ndarray] = []):
        self.arrival_times = arrival_times
        self.__ids__ = np.arange(0, len(arrival_times) + 1)[0]

    def append(self, arrival_time):
        if not isinstance(arrival_time, ArrivalTimeData):
            raise TypeError
        self.arrival_times.append(arrival_time)

    def __repr__(self):
        out_str = ''
        for arrival_time in self.arrival_times:
             out_str += f'{str(arrival_time)}\n'

        return out_str

    def __getitem__(self, items):
        if isinstance(items, (list, np.ndarray)):
            arrival_times = []
            for arrival_time in self.arrival_times:
                if arrival_time.id in list(items):
                    arrival_times.append(arrival_time)
            return ArrivalTimeEnsemble(arrival_times=arrival_times)

        elif isinstance(items, str):
            for arrival_time in self.arrival_times:
                if arrival_time.id == items:
                    return arrival_time

        else:
            raise TypeError

    def to_dict(self):
        out_dict = {'resource_id': [],
                    'id': [],
                    'event_id': [],
                    'site_id': [],
                    'site_name': [],
                    'arrival_time': [],
                    'phase': []}

        for k, arrival_time in enumerate(self.arrival_times):
            out_dict['resource_id'].append(arrival_time.id)
            out_dict['id'].append(k)
            out_dict['event_id'].append(arrival_time.event_id)
            out_dict['site_id'].append(arrival_time.site_id)
            out_dict['site_name'].append(arrival_time.site_name)
            out_dict['arrival_time'].append(arrival_time.arrival_time)
            out_dict['phase'].append(arrival_time.phase)

        return out_dict

    @property
    def event_ids(self):
        event_ids = [arrival_time.event_id for arrival_time
                     in self.arrival_times]

        return list(np.unique(event_ids))

    @property
    def site_ids(self):
        site_ids = [arrival_time.site_id for arrival_time
                     in self.arrival_times]

        return list(np.unique(site_ids))

    def groupby(self, column, **kwargs):
        df = pd.DataFrame(self.to_dict()).set_index('id')
        return df.groupby(column, **kwargs)


class TomoRay(Ray):
    def __init__(self, ray: Ray):
        self.__dict__ = {}
        for key in ray.__dict__.keys():
            self.__dict__[key] = ray.__dict__[key]

        dist = np.array([0] + list(np.linalg.norm(np.diff(self.nodes, 
                                                          axis=0),axis=1)))
        self.dist = np.cumsum(dist)
        
        self.int_x = sc.interpolate.interp1d(self.dist, self.nodes[:, 0])
        self.int_y = sc.interpolate.interp1d(self.dist, self.nodes[:, 1])
        self.int_z = sc.interpolate.interp1d(self.dist, self.nodes[:, 2])
        
        self.velocity = None
        self.epsilon = 10
        self.threshold = 0.1
        self.sensitivity_kernel = None
        
    def add_velocity(self, velocity: VelocityGrid3D):
        self.velocity = velocity
        x, y, z, v = velocity.flattens()
        self.vel_x = x
        self.vel_y = y
        self.vel_z = z
        self.flatten_vel = v
        
    def set_epsilon(self, epsilon):
        self.epsilon

    def sensitivity(self, distance_along_ray):
        """
        measures the sensitivity with respect to the velocity model parameters

        :NOTE: currently, the sensitivity assumes a rbf inverse distance
        interpolation kernel
        """

        from time import time
        t0 = time()
        location = self.interpolate_coordinate(distance_along_ray)

        result = self.rbf_interpolation_sensitivity(location)

        # result = self.velocity.rbf_interpolation_sensitivity(location,
        #                                                      self.epsilon,
        #                                                      threshold=
        #                                                      self.threshold)
        t1 = time()
        logger.info(f'dones measuring sensitivity in {t1 - t0}')

        return result

        # return result.reshape(np.prod(result.shape))
    
    def interpolate_coordinate(self, distance_along_ray):
        """
        return the coordinates at a given distance along the ray
        """

        if not (0 <= distance_along_ray <= np.max(self.dist)):
            raise ValueError(f'the value for "distance_along_ray" must be'
                             f'larger or equal to 0 and smaller or equal to '
                             f'{np.max(self.dist)}')

        return np.array([self.int_x(distance_along_ray),
                         self.int_y(distance_along_ray),
                         self.int_z(distance_along_ray)])

    def rbf_interpolation_sensitivity(self, location):
        """
        calculate the sensitivity of each element given a location
        :param location: location in model space at which the interpolation
        occurs
        :param epsilon: the standard deviation of the gaussian kernel
        :param threshold: threshold relative to the maximum value below which
        the weights are considered 0.
        :rparam: the sensitivity matrix
        """

        # calculating the distance between the location and every grid points

        dist = np.sqrt((self.vel_x - location[0]) ** 2 +
                       (self.vel_y - location[1]) ** 2 +
                       (self.vel_z - location[2]) ** 2)

        sensitivity = np.exp(-(dist / self.epsilon) ** 2)
        sensitivity[sensitivity < np.max(sensitivity) * self.threshold] = 0
        # sensitivity = sensitivity / np.sum(sensitivity)

        return sensitivity
    
    def integrate_sensitivity(self, velocity: VelocityGrid3D, epsilon=10,
                              threshold=0.1, max_it=100, compress_matrix=True):
        from time import time
        t0 = time()
        self.set_epsilon(epsilon)
        self.add_velocity(velocity)
        self.threshold = threshold

        y0 = np.zeros(np.prod(velocity.shape))
        slowness = 1 / velocity.data.reshape(y0.shape)
        sensitivity = y0
        for i, dl in enumerate(np.diff(self.dist)):
            distance_along_ray = self.dist[i] + dl
            sensitivity += self.sensitivity(distance_along_ray) * slowness * dl

        # sensitivity = sensitivity / np.sum(sensitivity) * self.travel_time

        if compress_matrix:
            sensitivity = csr_matrix(sensitivity)

        self.sensitivity_kernel = sensitivity
        t1 = time()
        logger.info(f'done calculating sensitivity in {t1 - t0:0.2f} seconds')

        return


class TomoRayEnsemble(BaseModel):
    def __init__(self, events: EventEnsemble, arrivals: ArrivalTimeEnsemble,
                 rays: List[Ray] = []):
        self.events = events
        self.arrivals = arrivals
        self.rays = rays

        self.site_ids = []
        for ray in rays:
            self.site_ids.append(ray.site_code)
            # for loc

    def append(self, ray: Ray):
        self.rays.append()


class Tomography(ProjectManager):
    def __init__(self, base_projects_path, project_name, network_code,
                 use_srces=False, solve_velocity=True, solve_location=True,
                 current_epoch=0, **kwargs):

        self.events = None
        self.observations = None
        self.rays = None
        self.solve_location = solve_location
        self.solve_velocity = solve_velocity
        self.epoch = current_epoch
        self.arrival_times = None

        super().__init__(base_projects_path, project_name, network_code,
                         use_srces=use_srces, **kwargs)

        self.paths.tomography = self.paths.root / 'tomography'

        if not self.paths.tomography.exists():
            self.paths.tomography.mkdir(parents=True, exist_ok=True)

        self.paths.current_epoch = self.paths.tomography / \
                                       f'epoch_{self.epoch}'

        self.paths.current_epoch.mkdir(parents=True, exist_ok=True)

    @property
    def travel_time_table_path(self):
        travel_time_table_path = self.paths.current_epoch / 'travel_times'
        travel_time_table_path.mkdir(parents=True, exist_ok=True)
        return travel_time_table_path

    def travel_time_table_file(self, site_name, phase):
        return self.travel_time_table_path \
               / f'travel_time_table_{site_name}_{phase}.pickle'

    def travel_time_table_files(self, phase):
        return np.sort([f for f in self.travel_time_table_path.glob(
                        f'*{phase}.pickle')])

    @property
    def current_epoch(self):
        return self.epoch

    @property
    def event_table_file(self):
        event_table_file = self.paths.current_epoch / 'event_table.pickle'
        event_table_file.parent.mkdir(parents=True, exist_ok=True)
        return event_table_file


    @property
    def site_table_file(self):
        site_table_file = self.paths.current_epoch / 'site_table.pickle'
        site_table_file.parent.mkdir(parents=True, exist_ok=True)
        return site_table_file

    @property
    def path_velocities(self):
        path_velocities = self.paths.current_epoch / 'velocities'
        path_velocities.mkdir(parents=True, exist_ok=True)
        return path_velocities

    def velocity_file(self, phase):
        return self.path_velocities / f'{phase}_velocity.pickle'

    @property
    def estuaire_path_prefix(self):
        return Path('/tmp')

    @property
    def travel_time_grids_path(self):
        path_travel_time_grids = self.paths.current_epoch / 'travel_time_grids'
        path_travel_time_grids.mkdir(parents=True, exist_ok=True)
        return path_travel_time_grids

    def travel_time_grid_file(self, site_name, phase):
        return self.travel_time_grids_path / \
               f'travel_time_grid_{site_name}_{phase}.pickle'

    @property
    def sensitivity_path(self):
        path_sensitivity = self.paths.current_epoch / 'sensitivities'
        path_sensitivity.mkdir(parents=True, exist_ok=True)
        return path_sensitivity

    def sensitivity_file(self, site_name, phase):
        return self.sensitivity_path / \
               f'sensitivity_{site_name}_{phase}.pickle'

    @property
    def site_table(self):
        site_list = [(site_id, list(site.loc), site.time_correction)
                     for site_id, site in enumerate(self.srces.sites)]

        site_list = np.array(site_list, dtype=estuaire_data.st_dtype)

        return estuaire_data.EKStationTable(data=site_list)

    @property
    def event_table(self):
        events = [(event_id, event.loc, event.origin_time_correction)
                  for event_id, event in enumerate(self.events)]

        events = np.array(events, dtype=estuaire_data.ev_dtype)

        return estuaire_data.EKEventTable(data=events)

    def travel_time_table(self, site_id: int, phase: Phase):
        ats = self.arrival_times.groupby('site_id').get_group(site_id)
        ttt = []
        ttt_id = 0
        for i, at in enumerate(ats.iterrows()):
            at = at[1]
            if at.phase != phase:
                continue
            travel_time = at.arrival_time - \
                self.events[at.event_id].origin_time
            ttt.append((ttt_id, at.event_id, travel_time.total_seconds()))
            ttt_id += 1

        ttt = np.array(ttt, dtype=estuaire_data.tt_dtype)

        event_table_file = self.estuaire_path_prefix / self.event_table_file
        site_table_file = self.estuaire_path_prefix / self.site_table_file

        tt_table = estuaire_data.EKTTTable(ttt, site_id,
                                           evnfile=event_table_file,
                                           stafile=site_table_file)

        return tt_table

    def travel_time_tables(self, phase: Phase):
        site_ids = [group[0] for group in
                    self.arrival_times.groupby('site_id')]
        site_names = [group[0] for group in
                      self.arrival_times.groupby('site_name')]

        tt_tables = [(site_name, self.travel_time_table(site_id, phase))
                     for site_id, site_name in
                     zip(site_ids, site_names)]

        return tt_tables

    def synthetic(self, dims=[100, 100, 100],
                  origin=[0, 0, 0],
                  spacing=[10, 10, 10],
                  p_mean=5000, p_std=200, s_mean=3000, s_std=125,
                  model_smoothing=3,
                  n_sites=20, n_events=1000,
                  min_observations=None, max_observations=None,
                  pick_perturbation_std=0.005,
                  synthetic_type: SyntheticTypes = SyntheticTypes('random'),
                  random_seed=None, multi_threaded=True):

        if synthetic_type == SyntheticTypes('random'):

            self.__add_random_velocities__(dims, origin, spacing,
                                           p_mean, p_std, s_mean, s_std,
                                           model_smoothing,
                                           random_seed=random_seed)

        else:
            raise ValueError('not implemented yet')

        self.__add_random_sites__(n_sites, random_seed=random_seed)
        self.__add_random_events__(n_events, random_seed=random_seed)
        self.init_travel_time_grids(multi_threaded=multi_threaded)
        self.__add_random_arrival_times__(min_observations=min_observations,
                                          max_observations=max_observations,
                                          p_pick_error=pick_perturbation_std,
                                          s_pick_error=pick_perturbation_std)

    def __add_random_velocities__(self, dims, origin, spacing,
                                  p_mean, p_std, s_mean, s_std,
                                  smoothing, random_seed=None,
                                  multi_threaded=True):

        p_velocity = VelocityGrid3D(self.network_code, dims, origin,
                                    spacing)

        s_velocity = VelocityGrid3D(self.network_code, dims, origin,
                                    spacing, phase='S')

        p_velocity.fill_random(p_mean, p_std, smoothing, random_seed)

        if random_seed is not None:
            random_seed += 10

        s_velocity.fill_random(s_mean, s_std, smoothing, random_seed)

        # self.add_velocity(p_velocity)
        # self.add_velocity(s_velocity)

        self.add_velocities(VelocityGridEnsemble(p_velocity, s_velocity),
                            initialize_travel_times=False)

    def __add_random_sites__(self, nstations, random_seed=None):
        """
        create random stations
        """

        if not self.has_p_velocity():
            logger.info('The project does not contain a p wave velocity...'
                        ' exiting')
            return

        if not self.has_s_velocity():
            logger.info('The project does not contain a s wave velocity...'
                        ' exiting')
            return

        sta_locs = self.p_velocity.generate_random_points_in_grid(
            nstations, seed=random_seed)

        sites = [Site(label=f'STA{i:02d}', x=sta_loc[0], y=sta_loc[1],
                      z=sta_loc[2]) for i, sta_loc in enumerate(sta_locs)]

        self.add_srces(Srces(sites), initialize_travel_time=False)

    def __add_random_events__(self, n_events, random_seed=None):
        if not self.has_p_velocity():
            logger.info('The project does not contain a p wave velocity...'
                        ' exiting')
            return

        if not self.has_s_velocity():
            logger.info('The project does not contain a s wave velocity...'
                        ' exiting')
            return

        origin_time = datetime.now()

        events = self.p_velocity.\
            generate_random_points_in_grid(n_events, seed=random_seed)

        self.events = EventEnsemble()
        for event in events:
            self.events.append(EventData(location=list(event),
                                         origin_time=origin_time))

        return self.events

    def __add_random_arrival_times__(self, min_observations=None,
                                     max_observations=None,
                                     p_pick_error=0.001,
                                     s_pick_error=0.001):
        """
        generate random travel time observations.
        :param min_observations: minimum number of observations for each
        event
        :param max_observations: maximum number of observations for each event
        :param p_pick_error: standard deviation of the gaussian
        perturbation in second to add to the travel time for the p picks
        :param s_pick_error: standard deviation of the gaussian
        perturbation in second to add to the travel time for the s picks
        """

        if min_observations is None:
            min_observations = 0
        elif min_observations < 0:
            min_observations = 0

        if max_observations is None:
            max_observations = len(self.srces)

        if max_observations > len(self.srces):
            max_observations = len(self.srces)

        origin_time = datetime.now()
        self.arrival_times = ArrivalTimeEnsemble()
        perturbation = {'P': p_pick_error,
                        'S': s_pick_error}
        i = 0
        for k, event_id in enumerate(self.events.dict.keys()):
            arrivals = []
            n_observations = np.random.randint(min_observations,
                                               max_observations)
            for sensor_id, sensor in enumerate(self.srces):
                for phase in ['P', 'S']:
                    travel_time = self.travel_times.select(
                        sensor.label, phase=phase)[0].interpolate(
                        self.events.dict[event_id].loc)[0]

                    travel_time += np.random.randn() * perturbation[phase]
                    pick_time = origin_time + timedelta(seconds=travel_time)

                    arrival = ArrivalTimeData(event_id=k,
                                              site_name=sensor.label,
                                              site_id=sensor_id,
                                              arrival_time=pick_time,
                                              phase=phase,
                                              resource_id=event_id.id)

                    arrivals.append(arrival)

            for arrival in np.random.choice(arrivals, n_observations,
                                            replace=False):
                self.arrival_times.append(arrival)

    def __write_site_table_file__(self):
        with open(self.site_table_file, 'wb') as site_table_file:
            pickle.dump(self.site_table, site_table_file)
        logger.info(f'writing the site table file ({self.site_table_file}')

    def __write_event_table_file__(self):
        with open(self.event_table_file, 'wb') as event_table_file:
            pickle.dump(self.event_table, event_table_file)
        logger.info(f'writing the event table file ({self.event_table_file}')

    def __write_travel_time_tables__(self):
        for phase in Phase:
            for site_name, tt_table in self.travel_time_tables(phase):
                with open(self.travel_time_table_file(site_name, phase),
                          'wb') as travel_time_table_file:

                    pickle.dump(tt_table, travel_time_table_file)
                logger.info(f'writing the travel time table file '
                            f'({self.travel_time_table_file(site_name, phase)}'
                            f')')

    def __write_velocity_grids__(self):
        with open(self.velocity_file(Phase('P')), 'wb') as p_velocity_file:

            data = self.p_velocity.data
            spacing = self.p_velocity.spacing[0]
            origin = self.p_velocity.origin

            p_velocity = estuaire_data.EKImageData(data, spacing=spacing,
                                                   origin=origin)

            pickle.dump(p_velocity, p_velocity_file)
            logger.info(f'writing the p velocity file ({p_velocity_file})')

        with open(self.velocity_file(Phase('S')), 'wb') as s_velocity_file:
            data = self.p_velocity.data
            spacing = self.p_velocity.spacing[0]
            origin = self.p_velocity.origin

            s_velocity = estuaire_data.EKImageData(data, spacing=spacing,
                                                   origin=origin)

            pickle.dump(s_velocity, s_velocity_file)
            logger.info(f'writing the s velocity file ({s_velocity_file})')

    def __write_travel_time_grids__(self):
        for travel_time_grid in self.travel_times:
            phase = Phase(travel_time_grid.phase)
            site_name = travel_time_grid.seed_label
            file_name = self.travel_time_grid_file(site_name, phase)

            data = travel_time_grid.data
            spacing = travel_time_grid.spacing[0]
            origin = travel_time_grid.origin
            seeds = np.array([travel_time_grid.transform_to(
                travel_time_grid.seed)])

            grid = estuaire_data.EKImageData(data, spacing=spacing,
                                             origin=origin, seeds=seeds)

            with open(file_name, 'wb') as output_file:
                pickle.dump(grid, output_file)

            logger.info(f'writing the travel time grid for site {site_name} '
                        f'({file_name}')

    def initialize_inversion(self):
        self.__write_site_table_file__()
        self.__write_event_table_file__()
        self.__write_travel_time_tables__()
        self.__write_velocity_grids__()
        self.__write_travel_time_grids__()

    def run_sensitivity(self):
        docker_client = docker.from_env()

        docker_volume = {os.getcwd(): {'bind': str(self.estuaire_path_prefix),
                                       'mode': 'rw'}}

        self.paths.current_epoch / 'sensitivity'
        for site_name in self.travel_times.seed_labels:
            for phase in Phase:
                sensitivity_file = self.estuaire_path_prefix / \
                                   self.sensitivity_file(site_name, phase)
                travel_time_grid_file = self.estuaire_path_prefix / \
                    self.travel_time_grid_file(site_name, phase)
                travel_time_table_file = self.estuaire_path_prefix / \
                    self.travel_time_table_file(site_name, phase)

                cmd = f'sensitivity ' \
                      f'--output {sensitivity_file} ' \
                      f'--arrival {travel_time_grid_file} ' \
                      f'--velocity /tmp/{self.velocity_file(phase)} ' \
                      f'--traveltime {travel_time_table_file} ' \
                      f'--grid_id {0}'

                logger.info(f'running sensitivity using the following command '
                            f'\n{cmd}')
                docker_client.containers.run('jpmercier/estuaire', cmd,
                                             volumes=docker_volume,
                                             mem_limit='4g')



    @staticmethod
    def ray_tracer(velocity, data):
        travel_time_grid = data[0]
        arrival_id = data[1]
        loc = data[2]

        ray = travel_time_grid.ray_tracer(loc)
        ray.arrival_id = arrival_id
        # if ray.phase == 'P':
        #     velocity = self.p_velocity
        # else:
        #     velocity = self.s_velocity
        tomo_ray = TomoRay(ray)
        tomo_ray.integrate_sensitivity(velocity)

        return tomo_ray

    def ray_tracing(self, cpu_utilisation=0.9):
        """
        calculate the rays for every station event pair
        """
        num_threads = int(np.ceil(cpu_utilisation * __cpu_count__))

        arrival_time_grouped = self.arrival_times.groupby(
            ['site_id', 'phase'])

        rays = []
        for site_id in tqdm(self.arrival_times.site_ids):
            for phase in Phase:
                df = arrival_time_grouped.get_group((site_id, phase))

                tt = self.travel_times.select(site_id,
                                              phase=phase.value)[0]

                locs = self.events[df.event_id.values].locs
                events = self.events[df.event_id.values]
                arrival_ids = df.resource_id.values
                tts = [tt] * len(arrival_ids)

                data = [(travel_time, arrival_id, loc)
                        for travel_time, arrival_id, loc
                        in zip(tts, arrival_ids, locs)]

                if phase == 'P':
                    velocity = self.p_velocity
                else:
                    velocity = self.s_velocity

                ray_tracer = partial(self.ray_tracer, velocity)

                for d in data:
                    ray_tracer(d)

                with Pool(num_threads) as pool:
                    rays_tmp = list(tqdm(pool.imap(ray_tracer,
                                                   data),
                                         total=len(locs)))

                    for ray in rays_tmp:
                        rays.append(ray)

                return rays

            