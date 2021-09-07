import unittest
import shutil
import unittest
from uquake.grid.nlloc import VelocityGridEnsemble
from uquake.nlloc.nlloc import Srces
from loguru import logger
import numpy as np
from useis.services.nlloc import VelocityGrid3D, Srces
from useis.services.grid_server import Server
from useis.services.grid_client import GridClient
from uquake.core import read_inventory

root_dir = 'projects'
test_project = 'test_project'
test_network = 'TN'

grid_dim = [10, 10, 10]
grid_spacing = [1, 1, 1]
grid_origin = [0, 0, 0]


def create_velocity():
    p_velocity_grid = VelocityGrid3D(test_network,
                                     grid_dim,
                                     grid_origin,
                                     grid_spacing,
                                     phase='P', value=5000)
    s_velocity_grid = VelocityGrid3D(test_network,
                                     grid_dim,
                                     grid_origin,
                                     grid_spacing,
                                     phase='S', value=3000)

    return p_velocity_grid, s_velocity_grid


def create_srces():
    p_velocity = VelocityGrid3D(test_network,
                                grid_dim,
                                grid_origin,
                                grid_spacing,
                                phase='P', value=5000)
    srces = Srces.generate_random_srces_in_grid(p_velocity, n_srces=20)
    return srces


def velocity_grid_ensemble(p_velocity, s_velocity):

    return VelocityGridEnsemble(p_velocity, s_velocity)
    # nll.add_velocities(velocity_grids, initialize_travel_time=False)


class MyTestCase(unittest.TestCase):
    def test_add_velocity(self):
        gc = GridClient()
        from useis.processors.nlloc import NLLOC

        p_velocity, s_velocity = create_velocity()
        velocity_ensemble = velocity_grid_ensemble(p_velocity, s_velocity)

        gc.add_velocity_grid_3d(p_velocity, 'test')
        gc.add_velocity_grid_3d(s_velocity, 'test')

        self.assertTrue(True)

    def test_concert_site_to_from_pb(self):
        srces = Srces.from_srces(create_srces())
        srces_proto = srces.to_proto()
        srces_2 = Srces.from_proto(srces_proto)

        if srces.sites == srces_2.sites:
            self.assertTrue(True)

    # def test_add_inventory(self):
    #     gc = GridClient()
    #     inv = read_inventory('/home/jpmercier/Repositories/'
    #                          'Nanometrics/data/OSA_Light.xml',
    #                          xy_from_lat_lon=True)
    #
    #     try:
    #         gc.add_inventory(inv, project='test', network=test_network)
    #     except Exception as e:
    #         logger.error(e)
    #         self.assertTrue(False)
    #
    #     self.assertTrue(True)

    def test_add_srces(self):
        gc = GridClient()
        p_velocity, s_velocity = create_velocity()
        srces = Srces.generate_random_srces_in_grid(p_velocity,
                                                    n_srces=20)

        gc.add_srces(srces, test_project, test_network)

        self.assertTrue(True)

        # p_velocity_grid = VelocityGrid3D(test_network,
        #                                  grid_dim,
        #                                  grid_origin,
        #                                  grid_spacing,
        #                                  phase='P', value=5000)
        #
        # dims = p_velocity_grid.dims
        #
        # p_velocity_grid.data = np.random.randn(np.prod(dims)).reshape(dims)
        #
        # proto = p_velocity_grid.to_proto()
        # p_velocity_grid_2 = VelocityGrid3D.from_proto(proto)
        #
        # self.assertEqual(p_velocity_grid, p_velocity_grid_2)

    # def test_add_velocities(self):


if __name__ == '__main__':
    server = Server()
    server.start()
    unittest.main()
    server.stop()
