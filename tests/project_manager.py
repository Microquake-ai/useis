import shutil
import unittest
from useis.core.project_manager import ProjectManager
from utils import TestProjectManager
from uquake.grid.nlloc import VelocityGrid3D, VelocityGridEnsemble
from uquake.grid.base import Grid
from uquake.nlloc.nlloc import Srces
from glob import glob
from loguru import logger
from utils import TestUtils

tu = TestUtils()

# root_dir = 'projects'
# test_project = 'test_project'
# test_network = 'TN'
#
# grid_dim = [10, 10, 10]
# grid_spacing = [1, 1, 1]
# grid_origin = [0, 0, 0]


class TestPM(unittest.TestCase):

    def test_creation(self):
        pm = TestProjectManager()
        self.assertTrue(pm.paths.root.exists())
        shutil.rmtree(root_dir)

    def test_list_project(self):
        pm = ProjectManager(root_dir, test_project, test_network)
        projects = pm.list_active_projects()
        self.assertIn(test_project, projects)
        self.assertNotIn('archives', projects)
        shutil.rmtree(root_dir)

    def test_list_networks_project(self):
        pm = ProjectManager(root_dir, test_project, test_network)
        networks = pm.list_active_networks_project()
        self.assertIn(test_network, networks)
        shutil.rmtree(root_dir)

    def test_list_all_networks(self):
        pm = ProjectManager(root_dir, test_project, test_network)
        out_dict = pm.list_all_active_networks()
        self.assertIn(test_project, out_dict.keys())
        self.assertIn(test_network, out_dict[test_project])
        shutil.rmtree(root_dir)

    def test_add_velocity(self):
        p_velocity_grid = VelocityGrid3D(test_network, grid_dim, grid_origin,
                                         grid_spacing, phase='P', value=5000)
        s_velocity_grid = VelocityGrid3D(test_network, grid_dim, grid_origin,
                                         grid_spacing, phase='S', value=3000)

        pm = ProjectManager(root_dir, test_project, test_network)
        pm.add_velocity(p_velocity_grid, initialize_travel_time=False)

        logger.warning(f'{glob(str(pm.files.p_velocity) + ".hdr")}')

        self.assertTrue(len(glob(str(pm.files.p_velocity) + '.hdr')) > 0)
        pm.add_velocity(s_velocity_grid, initialize_travel_time=False)
        self.assertTrue(len(glob(str(pm.files.s_velocity) + '.hdr')) > 0)
        shutil.rmtree(root_dir)

    @staticmethod
    def create_add_velocity_grids():
        p_velocity_grid = VelocityGrid3D(test_network, grid_dim, grid_origin,
                                         grid_spacing, phase='P', value=5000)
        s_velocity_grid = VelocityGrid3D(test_network, grid_dim, grid_origin,
                                         grid_spacing, phase='S', value=3000)

        pm = ProjectManager(root_dir, test_project, test_network)
        velocity_grids = VelocityGridEnsemble(p_velocity_grid, s_velocity_grid)
        pm.add_velocities(velocity_grids)

        return pm

    def test_add_velocity_ensemble(self):
        pm = self.create_add_velocity_grids()
        self.assertTrue(len(glob(str(pm.files.p_velocity) + '.hdr')) > 0)
        self.assertTrue(len(glob(str(pm.files.s_velocity) + '.hdr')) > 0)
        shutil.rmtree(root_dir)

    def test_add_inventory(self):
        grid = Grid(grid_dim, grid_origin, grid_spacing, value=5000)
        sites = Srces.generate_random_srces_in_grid(grid, n_srces=10)
        pm = ProjectManager(root_dir, test_project, test_network)
        pm.add_srces(sites, initialize_travel_time=False)
        self.assertTrue(pm.files.srces.exists())

    def test_generate_tt_grids(self):
        pm = self.create_add_velocity_grids()
        sites = Srces.generate_random_srces_in_grid(pm.p_velocity)
        pm.add_srces(sites)

        self.assertTrue(len(glob(str(
            pm.paths.times) + '/*')) > 0)


if __name__ == '__main__':
    unittest.main()
