from useis.core.project_manager import ProjectManager
from useis.processors.nlloc import NLLOC
from uquake.grid.nlloc import VelocityGrid3D, VelocityGridEnsemble
from uquake.nlloc.nlloc import Srces, Observations
import shutil
# import numpy as np


class TestNLLOC(NLLOC):

    base_projects_path = 'projects'
    project_name = 'test'
    network_code = 'TN'
    grid_dimensions = [100, 100, 100]
    grid_spacing = [10, 10, 10]
    grid_origin = [0, 0, 0]

    def __init__(self):
        super().__init__(self.base_projects_path, self.project_name,
                         self.network_code)

    @classmethod
    def create_test_project(cls):
        """
        Preparing a project with velocity grids, sensors (srces) and
        observations for testing.
        :return:
        """

        tpm = cls()

        p_velocity_grid = VelocityGrid3D(cls.network_code,
                                         cls.grid_dimensions,
                                         cls.grid_origin,
                                         cls.grid_spacing,
                                         phase='P', value=5000)
        s_velocity_grid = VelocityGrid3D(cls.network_code,
                                         cls.grid_dimensions,
                                         cls.grid_origin,
                                         cls.grid_spacing,
                                         phase='S', value=3000)

        velocity_grids = VelocityGridEnsemble(p_velocity_grid, s_velocity_grid)
        tpm.add_velocities(velocity_grids)

        srces = Srces.generate_random_srces_in_grid(p_velocity_grid,
                                                    n_srces=10)
        tpm.add_srces(srces)
        tpm.init_travel_time_grids()

        return tpm

    def rm(self):
        shutil.rmtree(self.root_dir)

    def run_location(self):
        observations = Observations.generate_random_observations_in_grid(
            self.travel_times)
        super().run_location(observations=observations)
        # observations = None, calculate_rays = True,
        # delete_output_files = True, event = None,
        # evaluation_mode: str = 'automatic',
        # evaluation_status: str = 'preliminary',
        # multithreading = False




