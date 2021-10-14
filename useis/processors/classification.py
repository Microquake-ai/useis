from ..core.project_manager import ProjectManager
import pickle
from pathlib import Path
from ..classifier.model import EventClassifier


class Classifier(ProjectManager):
    def __init__(self, base_projects_path: Path, project_name: str,
                 network_code: str, use_srces: bool=False):

        """
        Object to control the classification
        :param base_projects_path:
        :param project_name:
        :param network_code:
        :param use_srces:
        """

        self.event_classifier = None

        if self.files.classification_model.is_file():
            self.event_classifier = EventClassifier.read(
                self.files.classification_model)

        super().__init__(base_projects_path, project_name, network_code,
                         use_srces=use_srces)

    def add_model(self, model: EventClassifier):
        model.write(self.files.classification_model)

