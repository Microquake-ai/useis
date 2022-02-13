from ..core.project_manager import ProjectManager


class CatalogManager(ProjectManager):
    def __init__(self, base_projects_path, project_name, network_code,
                 **kwargs):

        super().__init__(base_projects_path, project_name, network_code)

        self.paths['catalog'] = self.base_projects_path / catalog
        self.paths['event_waveforms'] = self.paths.catalog / 'waveform'
        self.paths['event_quakemls'] = self.paths.catalog / 'quakeml'

        self.files['catalog_index'] = self.paths.
