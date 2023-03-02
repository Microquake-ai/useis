from ..core.project_manager import ProjectManager


class FileManager(ProjectManager):
    def __init__(self, base_projects_path, project_name, network_code, waveform_path='',
                 quakeml_path='', init_catalogue=True, **kwargs):

        super().__init__(base_projects_path, project_name, network_code)

        self.paths['waveforms'] = waveform_path
        self.paths['quakeml'] = quakeml_path

        self.paths['catalog'] = self.base_projects_path / catalog





        # self.files['catalog_index'] = self.paths.
