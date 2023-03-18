from ..core.project_manager import ProjectManager
import pickle
from pathlib import Path
from ..ai.model import EventClassifier
import uquake
from uquake.core.stream import Stream, Trace
from uquake.core import UTCDateTime
import os
import shutil
from ..settings.settings import Settings
from ..ai.database import DBManager
from uquake.core.logging import logger
from ..ai.model import generate_spectrogram
from ..ai.database import Record
import numpy as np
import matplotlib.pyplot as plt
from uquake.core.event import AttribDict
from ..ai.event_type_lookup import event_type_lookup


class Classifier(ProjectManager):
    def __init__(self, base_projects_path: Path, project_name: str,
                 network_code: str, use_srces: bool=False, reset_training=False,
                 sampling_rates=[3000, 4000, 6000], window_lengths=[1, 3, 10]):

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
        self.databases.training_database = f'sqlite:///{self.files.training_database}'

        if reset_training:
            # delete the database file
            if self.files.training_database.exists():
                self.files.training_database.unlink()
            for f in self.paths.training_dataset.glob('*/*'):
                f.unlink()

        self.training_db_manager = DBManager(self.databases.training_database)

        for key in self.paths.keys():
            path = self.paths[key]
            logger.info(path)
            path.mkdir(parents=True, exist_ok=True)

        self.settings = Settings(str(self.paths.config))

        self.training_dataset_dict = {'files': [],
                                      'type': [],
                                      'magnitude': [],
                                      'event time': [],
                                      'window length': [],
                                      'original file': []}

        self.window_length_seconds = window_lengths
        self.sampling_rates = sampling_rates

    # def __del__(self):
    #     self.training_db_manager

    def add_model(self, model: EventClassifier):
        self.event_classifier = model
        model.write(self.files.classification_model)

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

    def build_training_data_set_list(self):
        self.files.training_dataset = \
            [f for f in self.paths.training_dataset.glob('*/*.png')
             if '_td' not in str(f)]

        for f in self.files.training_dataset:
            pass

    def spectrogram(self, stream: Stream):
        return generate_spectrogram(stream)

    def create_spectrogram_training(self, stream: Stream, event_type: str,
                                    original_event_type: str,
                                    event_id: str, end_time: UTCDateTime,
                                    magnitude: float, expect_impulsive=True,
                                    override_exist=False, simulate_magnitude=False,):

        init_sampling_rate = stream[0].stats.sampling_rate
        # from ipdb import set_trace
        # set_trace()
        if not simulate_magnitude:
            sampling_rates = [init_sampling_rate]
        else:
            sampling_rates = self.sampling_rates

        starttime = stream[0].stats.starttime
        end_time_sample = (end_time - starttime) * init_sampling_rate
        for sampling_rate in sampling_rates:
            synthetic = False
            magnitude_change = np.log(init_sampling_rate / sampling_rate) / np.log(3)
            correction = (1 - init_sampling_rate / sampling_rate) * 0.8
            if magnitude == -999:
                magnitude = -1
            new_magnitude = magnitude + magnitude_change
            if magnitude_change == 0:
                synthetic = True
            end_time_resampled = starttime + end_time_sample / sampling_rate + correction
            for window_length in self.window_length_seconds:
                trs_resampled = []
                # tmp = stream.copy().composite()
                # stream.traces.append(tmp[0])
                for tr in stream.copy():
                    # if expect_impulsive:
                    #     if not test_impulsiveness(tr):
                    #         continue
                    tr_resampled = tr.copy()
                    tr_resampled.stats.sampling_rate = sampling_rate
                    trs_resampled.append(tr_resampled)

                    tr_trimmed = tr_resampled.trim(starttime=end_time_resampled
                                                             - window_length,
                                                   endtime=end_time_resampled)

                    st_trimmed = Stream(traces=[tr_trimmed])

                    spectrograms = generate_spectrogram(st_trimmed)

                    filenames = self.write_spectrogram(spectrograms, event_type,
                                                       end_time_resampled,
                                                       new_magnitude, window_length)

                    channel_code = tr_resampled.stats.channel

                    if 'GP' in channel_code:
                        sensor_type = 'geophone'
                    else:
                        sensor_type = 'accelerometer'

                    sfn = filenames[0]
                    oet = event_type_lookup[original_event_type]

                    self.training_db_manager.add_record(event_id=event_id,
                                                        spectrogram_filename=sfn,
                                                        channel_id=tr.stats.channel,
                                                        magnitude=new_magnitude,
                                                        duration=window_length,
                                                        end_time=str(end_time),
                                                        sampling_rate=sampling_rate,
                                                        categories=event_type,
                                                        original_event_type=oet,
                                                        mseed_file=event_id,
                                                        sensor_type=sensor_type,
                                                        bounding_box=[0, 0],
                                                        synthetic=synthetic)

                    # add_record(self, event_id, spectrogram_filename, channel_id,
                    #            magnitude,
                    #            sampling_rate, categories, mseed_file, sensor_type,
                    #            bounding_box):
                    for filename, tr in zip(filenames, st_trimmed):
                        plt.clf()
                        plt.plot(tr.data, 'k')
                        plt.axis('off')
                        plt.tight_layout()
                        filename = Path(filename)
                        classifier_dataset_path = self.paths.training_dataset
                        plt.savefig(
                            f'{classifier_dataset_path / event_type / filename.stem}'
                            f'_td.png')

                if not trs_resampled:
                    continue

                # from ipdb import set_trace
                # set_trace()

    def write_spectrogram(self, spectrograms, event_type, end_time, magnitude, duration):

        filenames = []
        for i, spectrogram in enumerate(spectrograms):
            if np.mean(spectrogram) == 0:
                continue

            filename = f'{event_type}_{end_time}_mag{magnitude:0.2f}_' \
                       f'{duration}sec_ch{i}.png'
            data_path = self.paths.training_dataset / event_type
            data_path.mkdir(parents=True, exist_ok=True)
            path = self.paths.training_dataset / event_type / filename
            spectrogram.save(path, format='png')
            filenames.append(filename)

        return filenames

    def event_exists(self, event_id):
        return self.training_db_manager.event_exists(event_id)


