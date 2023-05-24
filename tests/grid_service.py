import unittest
import shutil
import unittest
from uquake.grid.nlloc import VelocityGridEnsemble
from loguru import logger
import numpy as np
from useis.services.models.grid import VelocityGrid3D
from useis.services.grid.server import Server

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


def velocity_grid_ensemble(p_velocity, s_velocity):

    return VelocityGridEnsemble(p_velocity, s_velocity)
    # nll.add_velocities(velocity_grids, initialize_travel_time=False)


class MyTestCase(unittest.TestCase):
    def test_convert_to_pb(self):
        from useis.processors.nlloc import NLLOC

        p_velocity, s_velocity = create_velocity()
        velocity_ensemble = velocity_grid_ensemble(p_velocity, s_velocity)

        p_velocity_grid = VelocityGrid3D(test_network,
                                         grid_dim,
                                         grid_origin,
                                         grid_spacing,
                                         phase='P', value=5000)

        dims = p_velocity_grid.dims

        p_velocity_grid.data = np.random.randn(np.prod(dims)).reshape(dims)

        proto = p_velocity_grid.to_proto()
        p_velocity_grid_2 = VelocityGrid3D.from_proto(proto)

        self.assertEqual(p_velocity_grid, p_velocity_grid_2)

    # def test_add_velocities(self):


if __name__ == '__main__':
    server = Server()
    server.start()
    unittest.main()
    server.stop()
