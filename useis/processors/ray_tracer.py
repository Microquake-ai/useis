from ..core.project_manager import ProjectManager


class RayTracer(ProjectManager):
    """
    Ray tracing module
    """

    def __init__(self, base_projects_path, project_name, network_code,
                 **kwargs):
        super().__init__(base_projects_path, project_name, network_code,
                         **kwargs)
        pass
    