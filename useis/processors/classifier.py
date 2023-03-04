from ..core.project_manager import ProjectManager
import pickle
from pathlib import Path
from ..ai.model import EventClassifier
import uquake
from uquake.core.stream import Stream, Trace
import os
import shutil
from ..settings.settings import Settings
from ..ai.database import DBManager
from uquake.core.logging import logger
from ..ai.model import generate_spectrogram
from ..ai.database import Record


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

        self.files.classifier_settings = self.paths.config / 'classifier_settings.toml'

        if not self.files.classifier_settings.is_file():
            settings_template = Path(os.path.realpath(__file__)).parent / \
                                '../settings/classifier_settings_template.toml'

            shutil.copyfile(settings_template,
                            self.files.classifier_settings)

            super().__init__(base_projects_path, project_name, network_code)

        if self.files.classification_model.is_file():
            self.event_classifier = EventClassifier.read(
                self.files.classification_model)

        self.files.training_database = self.paths.databases / \
                                       'classifier_training.sqlite'
        self.paths.training_dataset = self.paths.root / 'classifier_training_files'

        self.training_db_manager = DBManager(self.paths.databases /
                                             self.files.training_database)

        for key in self.paths.keys():
            path = self.paths[key]
            logger.info(path)
            path.mkdir(parents=True, exist_ok=True)

        self.settings = Settings(str(self.paths.config))

    def add_model(self, model: EventClassifier):
        self.event_classifier = model
        model.write(self.files.classification_model)

    # def create_training_record(self, stream: uquake.core.stream.Stream,
    #                            category: str, bounding_box):
    #     window_lengths = np.array(classifier_project.settings.classifier.window_lengths)
    #     sampling_rate = stream[0].stats.sampling_rate
    #
    #     for i, tr in enumerate(stream):
    #
    #
    #
    #
    #         record = [Record(filename)
    #
    #             class Record(Base):
    #                 __tablename__ = 'records'
    #                 id = Column(Integer, primary_key=True)
    #                 image10s_filename = Column(String)
    #                 image3s_filename = Column(String)
    #                 image1s_filename = Column(String)
    #                 sampling_rate = Column(Float)
    #                 categories = Column(String)
    #                 event_id = Column(Integer)
    #                 bounding_box_start = Column(Integer)
    #                 bounding_box_end = Column(Integer)
    #
    #         image_files.append(file_name)

    def trace2spectrograms(self, trace: Trace):
        spectrograms = []
        for window_length in window_lengths:
            st = Stream(traces=[trace])

            start_time = end_time - window_length
            st_trimmed = st.trim(starttime=start_time, endtime=end_time,
                                 pad=True, fill_value=0)

            spectrogram = generate_spectrogram(st_trimmed)

            training_data_path = self.settings.paths.training_dataset

            channel = tr.stats.channel

            filename = f'{category}_{end_time}_ch_{channel}_sr_{sampling_rate}' \
                       f'_{window_lengths}_sec.png'
            path = training_data_path / filename
            spectrogram.save(path, format='png')


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


