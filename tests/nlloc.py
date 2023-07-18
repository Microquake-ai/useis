import shutil
import unittest
from uquake.grid.nlloc import VelocityGrid3D, VelocityGridEnsemble
from loguru import logger
import numpy as np

root_dir = 'projects'
test_project = 'test_project'
test_network = 'TN'

grid_dim = [10, 10, 10]
grid_spacing = [1, 1, 1]
grid_origin = [0, 0, 0]


class NLLOC(unittest.TestCase):
    def test_locate_event(self):
        from uquake.nlloc.nlloc import Srces, Observations
        from useis.processors.nlloc import NLLOC
        nll = NLLOC('projects', 'test', 'TN')

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

        velocity_grids = VelocityGridEnsemble(p_velocity_grid, s_velocity_grid)
        nll.add_velocities(velocity_grids, initialize_travel_times=False)

        srces = Srces.generate_random_srces_in_grid(p_velocity_grid,
                                                    n_srces=20)
        nll.add_srces(srces, initialize_travel_time=False)

        nll.init_travel_time_grids(multi_threaded=True)

        e_loc = nll.p_velocity.generate_random_points_in_grid()
        observations = Observations.generate_observations_event_location(
            nll.travel_times, e_loc=e_loc)

        result = nll.run_location(observations=observations)

        distance = np.linalg.norm(result.loc - e_loc)
        logger.info(f'\n{distance:0.1f} m - error\n'
                    f'{result.uncertainty * 2:0.1f} m - uncertainty (2 std) ')
        shutil.rmtree(root_dir)

        self.assertTrue(distance < 3 * result.uncertainty)


if __name__ == '__main__':
    unittest.main()
    shutil.rmtree(root_dir)
