from ..core.project_manager import ProjectManager
import pickle
from pathlib import Path
from ..ai.model import EventClassifier
import uquake
import os
import shutil
from ..settings.settings import Settings


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

        self.files.classifier_settings = self.paths.config / 'classifier_settings.py'

        if not self.files.classifier_settings.is_file():
            settings_template = Path(os.path.realpath(__file__)).parent / \
                                '../settings/classifier_settings_template.toml'

            shutil.copyfile(settings_template,
                            self.files.classifier_settings)

            super().__init__(base_projects_path, project_name, network_code)

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

    def train_model(self):
        pass

    def create_training_data_set(self):
        pass


