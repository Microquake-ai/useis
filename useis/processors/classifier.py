from ..core.project_manager import ProjectManager
import pickle
from pathlib import Path
from ..ai.model import EventClassifier
import uquake


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

        super().__init__(base_projects_path, project_name, network_code,
                         use_srces=use_srces)

        if self.files.classification_model.is_file():
            self.event_classifier = EventClassifier.read(
                self.files.classification_model)

    def add_model(self, model: EventClassifier):
        self.event_classifier = model
        model.write(self.files.classification_model)

    def add_model_from_file(self, filename: str):
        ec = EventClassifier.from_pretrained_model_file(filename)
        self.event_classifier = ec
        ec.write(self.files.classification_model)

    def predict(self, st: uquake.core.stream.Stream):
        """
        :param st: the waveforms
        :type st: uquake.core.stream.Stream
        :return:
        """
        return self.event_classifier.predict(st)


