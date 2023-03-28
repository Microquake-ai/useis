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
from importlib import reload
from ..ai import database as ai_database
from sklearn.model_selection import train_test_split
from ipdb import set_trace
from sqlalchemy import func
import random
from ..ai.dataset import ClassificationDataset
from tqdm import tqdm

reload(ai_database)


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

        self.training_db_manager = ai_database.DBManager(
            self.databases.training_database)

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
                                    override_exist=False, simulate_magnitude=False):
        """
        Creates spectrograms of a given stream and writes them to disk.

        :param stream: The input stream.
        :type stream: obspy.core.stream.Stream
        :param event_type: The type of event.
        :type event_type: str
        :param original_event_type: The original type of event.
        :type original_event_type: str
        :param event_id: The event ID.
        :type event_id: str
        :param end_time: The end time of the event.
        :type end_time: UTCDateTime
        :param magnitude: The magnitude of the event.
        :type magnitude: float
        :param expect_impulsive: Whether or not to expect an impulsive signal in the data.
        :type expect_impulsive: bool
        :param override_exist: Whether or not to override existing files.
        :type override_exist: bool
        :param simulate_magnitude: Whether or not to simulate the magnitude.
        :type simulate_magnitude: bool
        """
        # Get the initial sampling rate
        init_sampling_rate = stream[0].stats.sampling_rate

        # Set the sampling rates to either the initial rate or the set of sampling rates to simulate
        if not simulate_magnitude:
            sampling_rates = [init_sampling_rate]
        else:
            sampling_rates = self.sampling_rates

        # Get the start time
        starttime = stream[0].stats.starttime

        # Calculate the end time in samples
        end_time_sample = (end_time - starttime) * init_sampling_rate

        # Loop over each sampling rate
        for sampling_rate in sampling_rates:
            synthetic = False
            magnitude_change = np.log(init_sampling_rate / sampling_rate) / np.log(3)
            correction = (1 - init_sampling_rate / sampling_rate) * 0.8

            # If the magnitude is not given, set it to -1
            if magnitude == -999:
                magnitude = -1

            # Calculate the new magnitude
            new_magnitude = magnitude + magnitude_change

            # If the magnitude doesn't change, set synthetic to True
            if magnitude_change == 0:
                synthetic = True

            # Calculate the end time at the new sampling rate
            end_time_resampled = starttime + end_time_sample / sampling_rate + correction

            # Loop over each window length
            for window_length in self.window_length_seconds:
                trs_resampled = []

                # Loop over each trace in the stream
                for tr in stream.copy():
                    # if expect_impulsive:
                    #     if not test_impulsiveness(tr):
                    #         continue

                    # Resample the trace
                    tr_resampled = tr.copy()
                    tr_resampled.stats.sampling_rate = sampling_rate
                    trs_resampled.append(tr_resampled)

                    # Trim the trace
                    tr_trimmed = tr_resampled.trim(
                        starttime=end_time_resampled - window_length,
                        endtime=end_time_resampled)

                    # Create a new stream with the trimmed trace
                    st_trimmed = Stream(traces=[tr_trimmed])

                    # Generate spectrograms for the trimmed stream
                    spectrograms = generate_spectrogram(st_trimmed)

                    # Write the spectrograms to disk
                    filenames = self.write_spectrogram(spectrograms, event_type,
                                                       end_time_res)

                    # Get the channel code for the resampled trace
                    channel_code = tr_resampled.stats.channel

                    # Determine the sensor type based on the channel code
                    if 'GP' in channel_code:
                        sensor_type = 'geophone'
                    else:
                        sensor_type = 'accelerometer'

                    # Get the spectrogram filename, original event type, and other
                    # metadata
                    sfn = filenames[0]
                    oet = event_type_lookup[original_event_type]

                    # Add the record to the training database
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

                    # Save a thumbnail image of the spectrogram
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

                    # Continue to the next iteration if there are no resampled traces
                    if not trs_resampled:
                        continue

                # from ipdb import set_trace
                # set_trace()

    def write_spectrogram(self, spectrograms, event_type, end_time, magnitude, duration):
        """
        Writes spectrogram images to disk.

        :param spectrograms: A list of spectrograms to write to disk.
        :type spectrograms: list
        :param event_type: The type of event associated with the spectrograms.
        :type event_type: str
        :param end_time: The end time of the event.
        :type end_time: str
        :param magnitude: The magnitude of the event.
        :type magnitude: float
        :param duration: The duration of the event.
        :type duration: float
        :returns: A list of filenames corresponding to the saved spectrogram images.
        :rtype: list
        """
        # create an empty list to store the filenames
        filenames = []

        # loop over the spectrograms
        for i, spectrogram in enumerate(spectrograms):
            # skip empty spectrograms
            if np.mean(spectrogram) == 0:
                continue

            # create the filename for the spectrogram image
            filename = f'{event_type}_{end_time}' \
                       f'_mag{magnitude:0.2f}_{duration}sec_ch{i}.png'

            # create the directory structure for the spectrogram image
            data_path = self.paths.training_dataset / event_type
            data_path.mkdir(parents=True, exist_ok=True)

            # save the spectrogram image to disk
            path = self.paths.training_dataset / event_type / filename
            spectrogram.save(path, format='png')

            # add the filename to the list of filenames
            filenames.append(filename)

        # return the list of filenames
        return filenames

    def split_dataset(self, split_test=0.2, split_validation=0.1, use_synthetic=True):
        """
        Split the dataset into training, testing, and validation sets.

        Parameters:
            split_test (float): The proportion of the dataset to allocate to the test
            set.
            split_validation (float): The proportion of the dataset to allocate to the
            validation set.
            use_synthetic (bool): Whether to use synthetic data or not.

        Returns:
            A tuple containing the training set, the testing set, and the validation set.
        """

        # Filter the data by category and synthetic status

        session = self.training_db_manager.Session()

        wl = self.window_length_seconds[0]

        # df = self.training_db_manager.to_pandas()
        # df_gb = df.groupby(['event_id', 'channel_id'])

        if use_synthetic:
            seismic_events = self.training_db_manager.filter(categories='seismic event',
                                                             duration=wl)

            # seismic_events_groups = df[(df['categories'] == 'seismic event') &
            #                            (df['synthetic'] == False)].groupby(
            #     ['event_id', 'channel_id', 'end_time'])
        else:
            seismic_events = self.training_db_manager.filter(synthetic=False,
                                                             categories='seismic event',
                                                             duration=wl)
        blasts = self.training_db_manager.filter(categories='blast', duration=wl)
        noises = self.training_db_manager.filter(categories='noise', duration=wl)

        df = self.training_db_manager.to_pandas()

        # Determine the size of the smallest category
        n_samples = min(seismic_events.count(), blasts.count(), noises.count())

        # Randomly sample records from each category
        seismic_events_sample = seismic_events.order_by(func.random()).limit(
            n_samples).all()
        blasts_sample = blasts.order_by(func.random()).limit(n_samples).all()
        noises_sample = noises.order_by(func.random()).limit(n_samples).all()

        # Combine samples into one list
        combined_sample = seismic_events_sample + blasts_sample + noises_sample

        # Compute test and validation set sizes
        test_size = split_test + split_validation
        validation_size = split_validation / test_size

        # Split the data into training, testing, and validation sets
        train, test_validation = train_test_split(combined_sample, test_size=test_size,
                                                  random_state=42)
        test, validation = train_test_split(test_validation, test_size=validation_size,
                                            random_state=42)

        # creating the dataset object for the training

        def reorganize(record):
            record_out = []
            for r in record:
                file_path = self.paths.training_dataset / r.categories
                filenames = (file_path / r.spectrogram_filename,
                             file_path / r.spectrogram_filename.replace(
                                 f'{wl:d}sec', f'{self.window_length_seconds[1]:d}sec'),
                             file_path / r.spectrogram_filename.replace(
                                 f'{wl:d}sec', f'{self.window_length_seconds[2]:d}sec'))
                label = r.categories

                record_out.append({'s1': filenames[0],
                                   's2': filenames[1],
                                   's3': filenames[2],
                                   'label': label})
            return record_out

        return reorganize(train), reorganize(test), reorganize(validation)

    def event_exists(self, event_id):
        return self.training_db_manager.event_exists(event_id)

    @property
    def dataset(self):
        filelist
        return ClassificationDataset()




