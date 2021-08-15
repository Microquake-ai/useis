import shutil
import unittest
from useis.core.project_manager import ProjectManager
from uquake.grid.nlloc import VelocityGrid3D, VelocityGridEnsemble
from uquake.grid.base import Grid
from uquake.nlloc import Srces

root_dir = 'projects'
test_project = 'test_project'
test_network = 'test_network'

grid_dim = [10, 10, 10]
grid_spacing = [1, 1, 1]
grid_origin = [0, 0, 0]


class CreationArchiving(unittest.TestCase):

    def test_archiving(self):
        pm = ProjectManager(root_dir, test_project, test_network)
        pm.archive_network()
        self.assertFalse(pm.root_directory.exists())
        archive_dir = pm.archive_directory / test_project / test_network
        self.assertTrue(archive_dir)
        shutil.rmtree(root_dir)

    def test_unarchiving(self):
        pm = ProjectManager(root_dir, test_project, test_network)
        pm.archive_network()
        pm.unarchive_network()
        self.assertTrue(pm.root_directory.exists())
        time_dir = pm.root_directory / 'times'
        self.assertTrue(time_dir.exists())
        shutil.rmtree(root_dir)

    def test_creation(self):
        pm = ProjectManager(root_dir, test_project, test_network)
        self.assertTrue(pm.root_directory.exists())
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

    def test_list_all_archived_networks(self):
        pm = ProjectManager(root_dir, test_project, test_network)
        pm.archive_network()
        out_dict = pm.list_all_archived_networks()
        # from ipdb import set_trace
        # set_trace()
        self.assertIn(test_project, out_dict.keys())
        self.assertIn(test_network, out_dict[test_project])

        # shutil.rmtree(root_dir)

    def add_velocity(self):
        p_velocity_grid = VelocityGrid3D(test_network, grid_dim, grid_origin,
                                         grid_spacing, phase='P', value=5000)
        s_velocity_grid = VelocityGrid3D(test_network, grid_dim, grid_origin,
                                         grid_spacing, phase='S', value=3000)

        pm = ProjectManager(root_dir, test_project, test_network)
        pm.add_velocity(p_velocity_grid)

        self.assertTrue(pm.p_velocity_file.exists())
        pm.add_velocity(s_velocity_grid)
        self.assertTrue(pm.s_velocity_file.exists())
        shutil.rmtree(root_dir)

    def add_velocity_ensemble(self):
        p_velocity_grid = VelocityGrid3D(test_network, grid_dim, grid_origin,
                                         grid_spacing, phase='P', value=5000)
        s_velocity_grid = VelocityGrid3D(test_network, grid_dim, grid_origin,
                                         grid_spacing, phase='S', value=3000)

        pm = ProjectManager(root_dir, test_project, test_network)
        velocity_grids = VelocityGridEnsemble(p_velocity_grid, s_velocity_grid)
        pm.add_velocities(velocity_grids)
        self.assertTrue(pm.p_velocity_file.exists())
        self.assertTrue(pm.s_velocity_file.exists())
        shutil.rmtree(root_dir)

    def add_inventory(self):
        grid = Grid(grid_dim, grid_origin, grid_spacing, value=5000)
        locs = grid.generate_random_points_in_grid(10)












if __name__ == '__main__':
    unittest.main()
