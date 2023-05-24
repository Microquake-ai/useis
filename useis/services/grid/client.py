import requests
import pickle
from uquake.grid.nlloc import VelocityGrid3D
from uquake.core.inventory import Inventory
from uquake.nlloc.nlloc import Srces, Observations
from io import BytesIO
from furl import furl
from typing import Optional, List
import json
import numpy as np
from useis.services.models.grid import VelocityGrid3D as ModelVelocityGrid3D
from useis.services.models.nlloc import Observations as ModelObservations
import json


def serialize_object(obj, name):
    f_out = BytesIO()
    pickle.dump(obj, f_out)
    f_out.seek(0)
    f_out.name = name
    return f_out


class GridClient:
    def __init__(self, base_url, project, network, user=None, password=None):
        self.base_url = furl(base_url)
        self.project = project
        self.network = network
        self.user = user
        self.password = password

    def add_3d_velocity(self, velocity: VelocityGrid3D,
                        initialize_travel_times=False):
        url = self.base_url / 'velocity' / self.project / self.network \
              / 'add' / '3D'

        data = {'initialize_travel_times': initialize_travel_times}

        velocity_grid = ModelVelocityGrid3D.from_uquake(velocity)
        params = {'velocity': velocity_grid,
                  'initialize_travel_times': initialize_travel_times}

        return requests.post(url, params=data, json=velocity_grid.to_dict())

    def add_inventory(self, inventory: Inventory,
                      initialize_travel_times=False):
        f_out = BytesIO()
        inventory.write(f_out)
        f_out.seek(0)
        f_out.name = 'inventory'
        url = self.base_url / 'inventory' / self.project / self.network
        files = {'inventory': f_out}
        data = {'initialize_travel_times': initialize_travel_times}
        return requests.post(url, files=files, data=data)

    def add_srces(self, srces: Srces, initialize_travel_times=False):
        f_out = serialize_object(srces, 'srces')
        url = self.base_url / 'srces' / self.project / self.network
        files = {'srces': f_out}
        data = {'initialize_travel_times': initialize_travel_times}
        return requests.post(url, files=files, data=data)

    def add_srces(self, srces: Srces, initialize_travel_times=False):
        f_out = serialize_object(srces, 'srces')
        url = self.base_url / 'srces' / self.project / self.network
        files = {'srces': f_out}
        data = {'initialize_travel_times': initialize_travel_times}
        return requests.post(url, files=files, data=data)

    def initialize_travel_times(self, multi_threaded=False):
        data = {'multi_threaded': multi_threaded}
        url = self.base_url / 'travel_times' / self.project / self.network / \
              'init'
        return requests.get(url, data=data)

    def event_location(self, observations: Observations):
        url = self.base_url / 'event' / 'locate' / self.project / self.network
        model_observations = ModelObservations.from_uquake(observations)
        from ipdb import set_trace
        set_trace()
        return requests.post(url, json=model_observations.to_dict())

    def generate_random_points_in_grid(self, n_points: Optional[int] = 1):
        url = self.base_url / 'test' / self.project / self.network / \
              'random_locations'

        params = {'n_points': n_points}
        response = requests.get(url, params=params)
        flat_array = np.array(eval(json.loads(response.content)))
        n2 = int(len(flat_array) / n_points)
        flat_array = flat_array.reshape((n_points, n2))
        return response, flat_array

    def generate_random_observations(self, x, y, z):
        url = self.base_url / 'test' / self.project / self.network / \
            'observations'
        params = {'x': x, 'y': y, 'z': z}
        response = requests.get(url, params=params)
        model_observations = ModelObservations.parse_obj(
            json.loads(response.content))

        return response, model_observations.to_uquake()

    def list_3d_velocity(self):
        url = self.base_url / 'velocity' / self.project / self.network
        return requests.get(url)
