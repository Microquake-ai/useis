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
import dataclasses
from pydantic import BaseModel, conlist
from enum import Enum
from pydantic.typing import List, Union
from datetime import datetime
from uuid import uuid4
from functools import partial
import scipy as sc
from scipy.sparse import csr_matrix
from ..tomography import data as ekdata

__cpu_count__ = cpu_count()


class Phase(str, Enum):
    p = 'P'
    s = 'S'

    def __str__(self):
        return self.value

    def __expr__(self):
        return self.__str__()

    def __call__(self):
        return self.__str__()


class EventData(BaseModel):
    location: conlist(float, min_items=3, max_items=3)
    location_correction: conlist(float, min_items=3, max_items=3) = [0, 0, 0]
    resource_id: str=None
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resource_id = ResourceIdentifier()

    @property
    def loc(self):
        return np.array(self.location)

    @property
    def id(self):
        return self.resource_id.id

    def __str__(self):
        return f'{self.id},{self.location},{self.location_correction}'

    def __repr__(self):
        return str(self)
    

class EventEnsemble(object):
    def __init__(self, events: List[EventData] = []):
        self.events = events
        self.dict = {}
        self.ids = np.arange(0, len(events))
        for event in events:
            self.dict[event.id] = event
            
    def __len__(self):
        return len(self.events)
    
    def __getitem__(self, items):
        if isinstance(items, (list, np.ndarray)):
            events = []
            for item in items:
                if isinstance(item, str):
                    for event in self.events:
                        if event.id == item:
                            events.append(event)
                elif isinstance(item, int):
                    events.append(self.events[item])
                else:
                    raise TypeError
            return EventEnsemble(events=events)
        elif isinstance(items, str):
            for event in self.events:
                if event.id == items:
                    return EventEnsemble(events=[event])
        elif isinstance(items, int):
            return EventEnsemble(events=[self.events[items]])
        else:
            raise TypeError

    def __str__(self):
        out_str = ''
        for event in self.events:
            out_str += f'{event}\n'
        return out_str

    def __repr__(self):
        return str(self)

    def select(self, ids: Union[List[int], int]):
        return self[ids]
    
    def append(self, event: EventData):
        self.events.append(event)
        self.dict[event.id] = event

    @property
    def locs(self):
        locs = []
        for event in self.events:
            locs.append(event.location)
        return locs

    def to_ek_event_table_data(self):
        ev_data = [(event.resource_id, event_id, event.location,
                 event.location_correction) for event, event_id
                in zip(self.events, self.ids)]:
         data.EKEventTable()


class ArrivalTimeData(BaseModel):
    event_id: str
    site_id: str
    phase: Phase
    resource_id: str=None
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


class ArrivalTimeEnsemble():
    def __init__(self, arrival_times: List[ArrivalTimeData] = []):
        self.arrival_times = arrival_times

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
        out_dict = {'id': [],
                    'event_id': [],
                    'site_id': [],
                    'arrival_time': [],
                    'phase': [],
                    'resource_id': []}

        for arrival_time in self.arrival_times:
            out_dict['id'].append(arrival_time.id)
            out_dict['event_id'].append(arrival_time.event_id)
            out_dict['site_id'].append(arrival_time.site_id)
            out_dict['arrival_time'].append(arrival_time.arrival_time)
            out_dict['phase'].append(arrival_time.phase)
            out_dict['resource_id'].append(arrival_time.resource_id)

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

        # to calculate frechet just making sure the sensitivity is normalized,
        # the sensitivity is expressed in s (time)


        # integrator = sc.integrate.RK45(self.sensitivity, 0, y0, self.length)
        #
        # for i in range(0, max_it):
        #     integrator.step()
        #     if integrator.status == 'finished':
        #         break
        #
        # return integrator.y / np.sum(integrator.y)


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
                     **kwargs):
            
            self.events = None
            self.observations = None
            self.rays = None
            self.solve_location = solve_location
            self.solve_velocity = solve_velocity
            
            super().__init__(base_projects_path, project_name, network_code,
                             use_srces=use_srces, **kwargs)

            self.paths.tomography = self.paths.root / 'tomography'

            super().__init__(base_projects_path, project_name, network_code,
                             use_srces=use_srces, **kwargs)
            
        
        def synthetic(self, dims=[100, 100, 100], 
                      origin=[0, 0, 0], 
                      spacing=[10, 10, 10], 
                      p_mean=5000, p_std=200, s_mean=3000, s_std=125,
                      model_smoothing=3,
                      nsites=20, nevents=1000,
                      trave_time_completeness=0.7, perturbation=0.005,
                      random_seed=None, multi_threaded=True):
            
            self.__add_random_velocities__(dims, origin, spacing, 
                                           p_mean, p_std, s_mean, s_std,
                                           model_smoothing,
                                           random_seed=random_seed)
            self.__add_random_sites__(nsites, random_seed=random_seed)
            self.__add_random_events__(nevents, random_seed=random_seed)
            self.init_travel_time_grids(multi_threaded=multi_threaded)
            self.__add_random_travel_times__()


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
            
        def __add_random_events__(self, nevents, random_seed=None):
            if not self.has_p_velocity():
                logger.info('The project does not contain a p wave velocity...'
                            ' exiting')
                return

            if not self.has_s_velocity():
                logger.info('The project does not contain a s wave velocity...'
                            ' exiting')
                return
            
            events = self.p_velocity.\
                generate_random_points_in_grid(nevents, seed=random_seed)

            self.events = EventEnsemble()
            for id, event in enumerate(events):
                self.events.append(EventData(location=list(event)))
            
            return self.events
            
        def __add_random_travel_times__(self, completeness=0.6,
                                        min_observations=5,
                                        p_pick_error=0.001,
                                        s_pick_error=0.001):
            """
            generate random travel time observations.
            :param completeness: on average the number of observation per 
            event-sensor pair
            :param min_observations: minimum number of observation for each
            event
            :param p_pick_error: standard deviation of the gaussian
            perturbation in second to add to the travel time for the p picks
            :param s_pick_error: standard deviation of the gaussian
            perturbation in second to add to the travel time for the s picks
            """

            origin_time = UTCDateTime()
            self.arrival_times = ArrivalTimeEnsemble()
            perturbation = {'P': p_pick_error,
                            'S': s_pick_error}
            i = 0
            for event_id in self.events.dict.keys():
                arrivals = []
                for sensor in self.srces:
                    for phase in ['P', 'S']:
                        travel_time = self.travel_times.select(
                            sensor.label, phase=phase)[0].interpolate(
                            self.events.dict[event_id].loc)[0]

                        travel_time += np.random.randn() * perturbation[phase]
                        pick_time = origin_time + travel_time

                        arrival = ArrivalTimeData(event_id=event_id,
                                                  site_id=sensor.label,
                                                  arrival_time=pick_time,
                                                  phase=phase)

                        self.arrival_times.append(arrival)

                        i += 1

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


                
                



            
            