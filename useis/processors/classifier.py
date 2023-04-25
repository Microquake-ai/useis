from ..core.project_manager import ProjectManager
import pickle
from pathlib import Path
import uquake
from uquake.core.stream import Stream, Trace
from uquake.core import UTCDateTime, read, read_events
import os
import shutil
from ..settings.settings import Settings
from uquake.core.logging import logger
import numpy as np
import matplotlib.pyplot as plt
from uquake.core.event import AttribDict
from importlib import reload
from ..ai import database as ai_database
from ..ai import dataset
from ..ai.event_type_lookup import event_type_lookup
from useis.ai.model import generate_spectrogram, EventClassifier, EventClassifier2
from ..ai.database import Record, DBManager
from sklearn.model_selection import train_test_split
from ipdb import set_trace
from sqlalchemy import func
import random
from PIL import Image
from useis.ai import model
from torchvision import models
from uuid import uuid4
import torch
from torch.nn.functional import softmax
import useis
import json
from useis.services.models.classifier import ClassifierResults
import multiprocessing
from tqdm import tqdm
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

reload(ai_database)
reload(dataset)
reload(model)


class ClassifierResult(object):
    def __init__(self, raw_output, label_mapping, inputs, spectrograms, inventory,
                 event_location=None):
        self.raw_output = raw_output
        self.label_mapping = label_mapping
        self.inputs = inputs
        self.inventory = inventory
        self.event_location = event_location
        self.spectrograms = spectrograms

    @property
    def predictions(self):
        pred_classes = torch.argmax(self.probabilities, dim=1)

        # Convert the index tensor to a binary tensor
        binary_preds = torch.zeros_like(self.probabilities)
        binary_preds.scatter_(1, pred_classes.view(-1, 1), 1)

        return binary_preds

    @property
    def probabilities(self):
        return softmax(self.raw_output, dim=1)

    @property
    def predicted_classes(self):
        classes = []
        for prediction in self.predictions:
            for key in self.label_mapping.keys():
                if np.all(self.label_mapping[key] == prediction.numpy()):
                    break

            classes.append(key)

        return classes

    @property
    def raw_output_strings(self):
        output_strings = []
        for raw_output in self.raw_output.detach().numpy():
            output_string = ''
            indices = np.argsort(raw_output)[::-1]
            keys = [key for key in self.label_mapping.keys()]
            for j, i in enumerate(indices):
                output_string += f'{keys[i]}: {raw_output[i]:02.0f}'
                if j < 2:
                    # add divider
                    output_string += ' | '

            output_strings.append(output_string)
        return output_strings

    def predicted_class_ensemble(self, event_location=None):
        """
        Predicted class for the trace ensemble. Prediction from each individual trace is
        weighted with the inverse of the distance. Provides a probability for each class
        @param event_location:
        @return:
        """

        distances = []
        probabilities = {}
        for label in self.label_mapping.keys():
            probabilities[label] = 0
        if event_location is not None:
            for trace, prob in zip(self.inputs, self.probabilities.detach().numpy()):
                site = self.inventory.select(station=trace.stats.station,
                                             location=trace.stats.location).sites

                if len(site) == 0:
                    continue

                site_loc = site[0].loc

                distance = np.linalg.norm(site_loc - event_location)

                for i, label in enumerate(self.label_mapping.keys()):
                    probabilities[label] += prob[i] / distance

                distances.append(distance)

            for label in self.label_mapping.keys():
                probabilities[label] /= np.sum(1 / np.array(distances))
        else:
            nb_prob = 0
            for prob in self.probabilities.detach().numpy():
                for i, label in enumerate(self.label_mapping.keys()):
                    probabilities[label] += prob[i]
                nb_prob += 1

            for label in self.label_mapping.keys():
                probabilities[label] /= nb_prob

        return probabilities

    @property
    def class_count(self):
        str_out = ""
        for category in self.label_mapping.keys():
            str_out += f'{category}: ' \
                       f'{np.sum(np.array(self.predicted_classes) == category)}\n'

        print(str_out)

    def to_json(self):
        return json.dumps(self.to_dict())
        pass

    def to_fastapi(self):
        networks = []
        stations = []
        channels = []

        for tr in self.inputs:
            networks.append(tr.stats.network)
            stations.append(tr.stats.station)
            channels.append(tr.stats.channel)
        return ClassifierResults(networks=networks,
                                 stations=stations,
                                 channels=channels,
                                 raw_outputs=self.raw_output.detach().numpy().tolist(),
                                 predicted_classes=self.predicted_classes,
                                 probabilities=self.probabilities.detach().numpy(
                                 ).tolist())

    def to_dict(self):
        outputs = []
        raw_outputs = self.raw_output.detach().numpy()
        predicted_classes = self.predicted_classes
        probabilities = self.probabilities.detach().numpy()
        for i in range(0, len(raw_outputs)):
            outputs.append({'network': self.inputs[i].stats.network,
                            'station': self.inputs[i].stats.station,
                            'channel': self.inputs[i].stats.channel,
                            'raw_output': raw_outputs[i].tolist(),
                            'predicted class': predicted_classes[i],
                            'probability': probabilities[i].tolist()})

        return outputs

    @property
    def ensemble_result(self):
        ensemble_prob = self.predicted_class_ensemble(event_location=self.event_location)

        label = None
        label_probability = 0
        for key in ensemble_prob.keys():
            if ensemble_prob[key] > label_probability:
                label_probability = ensemble_prob[key]
                label = key

        return f'{label}: {label_probability * 100 :0.2f}%'

    def __repr__(self):
        str_out = f'{self.inputs[0].stats.starttime}\n'
        for tr, predicted_class, probability in zip(self.inputs, self.predicted_classes,
                                                    self.probabilities):
            probability = probability.detach().numpy()
            formatted_probabilities = [f"{p:.2e}" for p in probability]

            ensemble_prob = self.predicted_class_ensemble(
                event_location=self.event_location)

            str_out += f'{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.' \
                       f'{tr.stats.channel}, class: {predicted_class} ' \
                       f'({np.max(probability):0.2%})\n' \

        str_out += f'\nSummary -> {self.ensemble_result}\n'

        return str_out


        # softmax(self.event_classifier.model(merged_images).cpu(), dim=1)

    # @property
    # def results(self):


class Classifier(ProjectManager):
    def __init__(self, base_projects_path: Path, project_name: str,
                 network_code: str, use_srces: bool=False, reset_training=False,
                 sampling_rates=[3000, 4000, 6000], window_lengths=[1, 3, 10],
                 gpu: bool = True, load_model: bool = True):

        """
        Object to control the classification
        :param base_projects_path:
        :param project_name:
        :param network_code:
        :param use_srces:
        """

        self.event_classifier = None
        self.gpu = gpu

        super(Classifier, self).__init__(base_projects_path, project_name, network_code,
                                         use_srces=use_srces)

        self.files.classifier_settings = self.paths.config / 'classifier_settings.toml'

        if not self.files.classifier_settings.is_file():
            settings_template = Path(os.path.realpath(__file__)).parent / \
                                '../settings/classifier_settings_template.toml'

            shutil.copyfile(settings_template,
                            self.files.classifier_settings)

            super().__init__(base_projects_path, project_name, network_code)

        # if load_model:
        #     if self.files.classification_model.is_file():
        #         self.event_classifier = EventClassifier.read(
        #             self.files.classification_model)

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

        self.train_accuracy = None
        self.train_loss = None
        self.test_accuracy = None
        self.test_loss = None
        self.validation_accuracy = None
        self.validation_loss = None

        if load_model:
            if self.files.classification_model.exists():
                try:
                    self.event_classifier = EventClassifier.read(
                        self.files.classification_model, gpu=self.gpu)
                except Exception as e:
                    logger.error(e)

            else:
                logger.warning('The project does not contain a classification model\n'
                               'use the add_model function to add a classification '
                               'model')

    # def __del__(self):
    #     self.training_db_manager

    def add_model(self, classifier_model: useis.ai.model.EventClassifier):
        if not isinstance(classifier_model, useis.ai.model.EventClassifier):
            raise TypeError(f'classifier_model must be an instance of '
                            f'useis.ai.model.EventClassifier, '
                            f'got {type(classifier_model)} instead')
        self.event_classifier = classifier_model
        self.event_classifier.write(self.files.classification_model)

    def add_model_from_file(self, classifier_model_path: str):
        classifier_model = model.EventClassifier.read(classifier_model_path,
                                                      gpu=self.gpu)
        self.add_model(classifier_model)

    def model_migration(self):
        """
        This function should be used when the EventClassifier object changes to ensure
        it is up-to-date
        """
        new_ec = model.EventClassifier(self.num_classes, self.label_mapping)
        new_ec.model = self.event_classifier.model

        self.add_model(new_ec)

    def trace2spectrograms(self, trace: Trace):
        spectrograms = []
        for window_length in self.window_length_seconds:
            st = Stream(traces=[trace.copy()]).copy()
            end_time = trace.stats.endtime
            start_time = end_time - window_length
            st_trimmed = st.trim(starttime=start_time, endtime=end_time,
                                 pad=True, fill_value=0)

            spectrogram = generate_spectrogram(st_trimmed)

            # if save_file:
            #     filename = f'{category}_{end_time}_ch_{channel}_sr_{sampling_rate}' \
            #                f'_{window_lengths}_sec.png'
            #     path = training_data_path / filename
            #     spectrogram.save(path, format='png')
            spectrograms.append(spectrogram)
        return spectrograms

    def predict(self, st: uquake.core.stream.Stream, event_location=None):
        """
        :param st: the waveforms
        :type st: uquake.core.stream.Stream
        :return:
        """

        self.event_classifier.model.eval()
        images = []
        for tr in st.copy():
            specs = self.trace2spectrograms(tr.copy())
            image = dataset.SpectrogramDataset.merge_reshape(
                specs[0][0], specs[1][0], specs[2][0]).to(self.event_classifier.device)
            image = image.view(1, image.shape[0], image.shape[1], image.shape[2])

            images.append(image)

        merged_images = torch.cat(images, dim=0)

        return ClassifierResult(self.event_classifier.model(merged_images).cpu(),
                                self.label_mapping, st, image, self.inventory.copy(),
                                event_location=event_location)

    def build_training_data_set_list(self):
        self.files.training_dataset = \
            [f for f in self.paths.training_dataset.glob('*/*.png')
             if '_td' not in str(f)]

        for f in self.files.training_dataset:
            pass

    # @staticmethod
    # def spectrogram(stream: Stream):
    #     return generate_spectrogram(stream)

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
        :param expect_impulsive: Whether to expect an impulsive signal in the data.
        :type expect_impulsive: bool
        :param override_exist: Whether to override existing files.
        :type override_exist: bool
        :param simulate_magnitude: Whether to simulate the magnitude.
        :type simulate_magnitude: bool
        """
        # Get the initial sampling rate
        init_sampling_rate = stream[0].stats.sampling_rate

        # Set the sampling rates to either the initial rate or the set of sampling
        # rates to simulate
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
            if not isinstance(self.window_length_seconds, list):
                window_lengths = [self.window_length_seconds]
            else:
                window_lengths = self.window_length_seconds
            for window_length in window_lengths:
                trs_resampled = []

                # Loop over each trace in the stream
                for tr in stream.copy():

                    # Resample the trace
                    tr_resampled = tr.copy()
                    tr_resampled.stats.sampling_rate = sampling_rate
                    trs_resampled.append(tr_resampled)

                    # Trim the trace
                    tr_trimmed = tr_resampled.trim(
                        starttime=end_time_resampled - window_length,
                        endtime=end_time_resampled, pad=True, fill_value=0)

                    # Create a new stream with the trimmed trace
                    st_trimmed = Stream(traces=[tr_trimmed])
                    st_trimmed = st_trimmed.detrend('demean').detrend('linear').taper(
                        max_percentage=1, max_length=0.01
                    )

                    # Generate spectrograms for the trimmed stream
                    spectrograms = generate_spectrogram(st_trimmed)

                    # Write the spectrograms to disk
                    duration = end_time_resampled - starttime
                    filenames = self.write_spectrogram(spectrograms, event_type,
                                                       end_time_resampled, magnitude,
                                                       duration, tr.stats.station,
                                                       tr.stats.location,
                                                       tr.stats.channel)

                    # Get the channel code for the resampled trace
                    channel_code = tr_resampled.stats.channel

                    # Determine the sensor type based on the channel code
                    if 'GP' in channel_code:
                        sensor_type = 'geophone'
                    else:
                        sensor_type = 'accelerometer'

                    # Get the spectrogram filename, original event type, and other
                    # metadata
                    if len(filenames) == 0:
                        return
                    sfn = filenames[0]
                    oet = event_type_lookup[original_event_type]

                    # Add the record to the training database
                    self.training_db_manager.add_record(event_id=event_id,
                                                        spectrogram_filename=sfn,
                                                        station=tr.stats.station,
                                                        location=tr.stats.location,
                                                        channel=tr.stats.channel,
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

    def write_spectrogram(self, spectrograms, event_type, end_time, magnitude, duration,
                          station, location, channel):
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
                       f'_mag_{magnitude:0.2f}_{duration}sec_{station}_' \
                       f'{location}_{channel}.png'

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
            labels = []
            for r in record:
                file_path = self.paths.training_dataset / r.categories
                filenames = (file_path / r.spectrogram_filename,
                             file_path / r.spectrogram_filename.replace(
                                 f'{wl:d}sec', f'{self.window_length_seconds[1]:d}sec'),
                             file_path / r.spectrogram_filename.replace(
                                 f'{wl:d}sec', f'{self.window_length_seconds[2]:d}sec'))
                label = r.categories

                if filenames[0].exists() & filenames[1].exists() & filenames[2].exists():
                    record_out.append({'s1': filenames[0],
                                       's2': filenames[1],
                                       's3': filenames[2]})
                    labels.append(label)
            return record_out, labels

        self.train = dataset.SpectrogramDataset(*reorganize(train),
                                                labels_mapping=self.label_mapping)
        self.test = dataset.SpectrogramDataset(*reorganize(test),
                                               labels_mapping=self.label_mapping)
        self.validation = dataset.SpectrogramDataset(*reorganize(validation),
                                                     labels_mapping=self.label_mapping)

        return self.train, self.test, self.validation

    @property
    def label_mapping(self):
        if self.event_classifier is None:
            unique_labels = self.training_db_manager.categories
            label_mapping = {}
            for i, label in enumerate(unique_labels):
                label_mapping[label] = np.zeros(len(unique_labels))
                label_mapping[label][i] = 1
            return label_mapping
        else:
            return self.event_classifier.label_mapping

    @property
    def num_classes(self):
        return len(self.label_mapping)

    def get_label(self, probs):
        pred_classes = torch.argmax(probs, dim=1)

        # Convert the index tensor to a binary tensor
        binary_preds = torch.zeros_like(probs)
        binary_preds.scatter_(1, pred_classes.view(-1, 1), 1)
        # still some work to do here.

    def train(self, learning_rate: float = 0.002,
              starting_model=None, override_current_model=False,
              batch_size: int = 500, plot_progress: bool = True,
              model_id: str = None, save_intermediate_models: bool = False,
              save_final_model: bool = True, use_synthetic: bool = True):

        if model_id is None:
            model_id = str(uuid4())

        if (self.event_classifier is None) | override_current_model:
            if starting_model is None:
                logger.info('using the ResNet34 as the default starting model')
                starting_model = models.resnet34()
        else:
            starting_model = self.event_classifier.model

        train, test, validation = self.split_dataset(use_synthetic=use_synthetic)

        ec = model.EventClassifier(self.num_classes, self.label_mapping,
                                   learning_rate=learning_rate, model=starting_model)

        test_losses = []
        test_accuracies = []

        train_losses = []
        train_accuracies = []

        for epoch in tqdm(range(0, 100)):
            ec.train(train, batch_size=batch_size)
            accuracy, loss = ec.validate(test, batch_size=batch_size)

            logger.info(f'Average iteration loss (training): {np.mean(ec.losses): 0.3f} '
                        f'+/- {np.std(ec.losses): 0.3f}')
            logger.info(f'Average iteration loss (test): {np.mean(loss): 0.3f} '
                        f'+/- {np.std(loss): 0.3f}')

            logger.info(f'Average iteration accuracy (training): '
                        f'{np.mean(ec.accuracies):0.3f} '
                        f'+/- {np.std(ec.accuracies): 0.3f}')
            logger.info(f'Average iteration accuracy (test: {np.mean(accuracy):0.3f} '
                        f'+/- {np.std(accuracy): 0.3f}')

            test_losses.append(np.mean(loss))
            test_accuracies.append(np.mean(accuracy))

            train_losses.append(np.mean(ec.losses))
            train_accuracies.append(np.mean(ec.accuracies))
            # accuracies.append(np.mean(ec.accuracies))

            if (epoch + 1) % 10 == 0:
                # ec.model.eval()
                ec.optimizer.param_groups[0]['lr'] /= np.sqrt(2)
                if save_intermediate_models:
                    self.add_model(ec)

            if plot_progress:
                plt.figure(1)
                plt.clf()
                plt.ion()

                plt.xlabel('epoch')
                plt.ylabel('accuracy')

                plt.plot(test_accuracies, '.-', label='test accuracy')
                plt.plot(train_accuracies, '.-', label='train accuracy')

                plt.legend()
                plt.draw()
                plt.pause(0.1)
                plt.show()

        if save_final_model:
            self.add_model(ec)

        validation_accuracy, validation_loss = ec.validate(validation,
                                                           batch_size=batch_size)

        ec.validation_accuracy = validation_accuracy
        ec.validation_loss = validation_loss
        ec.model_id = model_id

        return ec, (train, test, validation), \
            (test_losses, test_accuracies, train_losses, train_accuracies)

    def event_exists(self, event_id):
        return self.training_db_manager.event_exists(event_id)


class Classifier2(Classifier):
    def __init__(self, base_projects_path: Path, project_name: str,
                 network_code: str, use_srces: bool = False, reset_training=False,
                 sampling_rates=[3000, 4000, 6000],
                 gpu: bool = True):

        super(Classifier2, self).__init__(base_projects_path, project_name, network_code,
                                          use_srces=use_srces,
                                          reset_training=reset_training,
                                          sampling_rates=sampling_rates, gpu=gpu,
                                          load_model=False)

        self.window_length_seconds = self.settings.classifier.window_lengths

        if self.files.classification_model.exists():
            self.event_classifier = EventClassifier2.read(
                self.files.classification_model, gpu=self.gpu)

        else:
            logger.warning('The project does not contain a classification model\n'
                           'use the add_model function to add a classification '
                           'model')

        self.image_width = self.settings.classifier.image_width
        self.image_height = self.settings.classifier.image_height
        self.buffer_image_fraction = self.settings.classifier.buffer_image_fraction
        self.sampling_rate = self.settings.classifier.sampling_rate
        self.sequence_length_seconds = self.settings.classifier.window_lengths[0]

        self.buffer_image_sample = int(self.image_width * self.buffer_image_fraction)

        self.hop_length = int(np.floor(self.sequence_length_seconds * self.sampling_rate /
                                  (self.image_width + 2 * self.buffer_image_sample)))

        mel_spec = MelSpectrogram(sample_rate=self.sampling_rate,
                                  n_mels=self.image_height,
                                  hop_length=self.hop_length,
                                  power=1,
                                  pad_mode='reflect',
                                  normalized=True)

        self.mel_spectrogram = mel_spec

        self.amplitude_to_db = AmplitudeToDB()

    def spectrogram(self, trace: Trace, trim_from_start=True):

        tr = trace.copy().resample(self.settings.classifier.sampling_rate)
        wl = self.window_length_seconds[0]

        specs = []

        if trim_from_start:
            starttime = trace.stats.starttime
            endtime = trace.stats.starttime + wl
        else:
            endtime = trace.stats.endtime
            starttime = trace.stats.starttime - wl

        tr.detrend('demean').taper(max_percentage=0.1, max_length=0.01)
        tr.trim(starttime=starttime, endtime=endtime, pad=True, fill_value=0)

        data = tr.data
        data -= np.mean(data)

        torch_data = torch.tensor(data).type(torch.float32)

        spec = (self.mel_spectrogram(torch_data))
        spec -= spec.min()
        spec = spec / np.abs(spec).max() * 255
        # spec /= spec.max()
        # spec_db = self.amplitude_to_db(spec.abs() + 1e-3)
        # spec_db = (spec_db - spec_db.min()).numpy()
        # spec_db = spec_db[:, self.buffer_image_sample:-self.buffer_image_sample]
        # if spec_db.shape[1] > self.image_width:
        #     spec_db = spec_db[:, spec_db.shape[1] - self.image_width:]
        #
        # spec_db = spec_db - spec_db.min()
        # spec_db = spec_db / np.abs(spec_db).max() * 255
        # set_trace()
        spec = Image.fromarray(np.array(spec.tolist()).astype(np.uint8))

        image = dataset.SpectrogramDataset2.transform(spec)
        image_tensor = image.unsqueeze(0)

        return image_tensor

    def spectrogram_stream(self, stream, trim_from_start=True):
        specs = []
        for tr in stream:
            spec = self.spectrogram(tr, trim_from_start=trim_from_start)
            specs.append(spec)
        specs = torch.cat(specs, dim=0)
        return specs

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
            labels = []
            for r in record:
                file_path = self.paths.training_dataset / r.categories
                filename = (file_path / r.spectrogram_filename)
                label = r.categories

                if filename.exists():
                    record_out.append({'s1': filename})
                    labels.append(label)
            return record_out, labels

        self.train = dataset.SpectrogramDataset2(*reorganize(train),
                                                 labels_mapping=self.label_mapping)
        self.test = dataset.SpectrogramDataset2(*reorganize(test),
                                                labels_mapping=self.label_mapping)
        self.validation = dataset.SpectrogramDataset2(*reorganize(validation),
                                                      labels_mapping=self.label_mapping)

        return self.train, self.test, self.validation

    def add_model(self, classifier_model: useis.ai.model.EventClassifier2):
        if not isinstance(classifier_model, useis.ai.model.EventClassifier2):
            raise TypeError(f'classifier_model must be an instance of '
                            f'useis.ai.model.EventClassifier, '
                            f'got {type(classifier_model)} instead')
        self.event_classifier = classifier_model
        self.event_classifier.write(self.files.classification_model)

    def add_model_from_file(self, classifier_model_path: str):
        classifier_model = model.EventClassifier2.read(classifier_model_path,
                                                       gpu=self.gpu)
        self.add_model(classifier_model)

    def model_migration(self):
        """
        This function should be used when the EventClassifier object changes to ensure
        it is up-to-date
        """
        new_ec = model.EventClassifier2(self.num_classes, self.label_mapping)
        new_ec.model = self.event_classifier.model

        self.add_model(new_ec)

    def train(self, learning_rate: float = 0.002,
              starting_model=None, override_current_model=False,
              batch_size: int = 500, plot_progress: bool = True,
              model_id: str = None, save_intermediate_models: bool = False,
              intermediate_model_saving_interval: int = 10,
              save_final_model: bool = True, use_synthetic: bool = True,
              nb_epoch: int = 100, weight_decay: float = 0.001,
              dropout_prob: float = 0.2):

        if model_id is None:
            model_id = str(uuid4())

        if (self.event_classifier is None) | override_current_model:
            if starting_model is None:
                logger.info('using the ResNet34 as the default starting model')
                starting_model = models.resnet34()
        else:
            starting_model = self.event_classifier.model

        train, test, validation = self.split_dataset(use_synthetic=use_synthetic)

        ec = model.EventClassifier2(self.num_classes, self.label_mapping,
                                    learning_rate=learning_rate,
                                    model=starting_model,
                                    weight_decay=weight_decay,
                                    dropout_prob=dropout_prob)

        test_losses = []
        test_accuracies = []

        train_losses = []
        train_accuracies = []

        for epoch in tqdm(range(0, nb_epoch)):
            ec.train(train, batch_size=batch_size)
            accuracy, loss = ec.validate(test, batch_size=batch_size)

            logger.info(f'Average iteration loss (training): {np.mean(ec.losses): 0.3f} '
                        f'+/- {np.std(ec.losses): 0.3f}')
            logger.info(f'Average iteration loss (test): {np.mean(loss): 0.3f} '
                        f'+/- {np.std(loss): 0.3f}')

            logger.info(f'Average iteration accuracy (training): '
                        f'{np.mean(ec.accuracies):0.3f} '
                        f'+/- {np.std(ec.accuracies): 0.3f}')
            logger.info(f'Average iteration accuracy (test: {np.mean(accuracy):0.3f} '
                        f'+/- {np.std(accuracy): 0.3f}')

            test_losses.append(np.mean(loss))
            test_accuracies.append(np.mean(accuracy))

            train_losses.append(np.mean(ec.losses))
            train_accuracies.append(np.mean(ec.accuracies))
            # accuracies.append(np.mean(ec.accuracies))

            if (epoch + 1) % intermediate_model_saving_interval == 0:
                # ec.model.eval()
                ec.optimizer.param_groups[0]['lr'] /= np.sqrt(2)
                if save_intermediate_models:
                    self.add_model(ec)

            if plot_progress:
                plt.figure(1)
                plt.clf()
                plt.ion()

                plt.xlabel('epoch')
                plt.ylabel('accuracy')

                plt.plot(test_accuracies, '.-', label='test accuracy')
                plt.plot(train_accuracies, '.-', label='train accuracy')

                plt.legend()
                plt.draw()
                plt.pause(0.1)
                plt.show()

        if save_final_model:
            self.add_model(ec)

        validation_accuracy, validation_loss = ec.validate(validation,
                                                           batch_size=batch_size)

        ec.validation_accuracy = validation_accuracy
        ec.validation_loss = validation_loss
        ec.model_id = model_id

        return ec, (train, test, validation), \
            (test_losses, test_accuracies, train_losses, train_accuracies)

    def create_training_dataset(self, data_path, reset_training=True):
        data_path = Path(data_path)
        mseed_files = [filename for filename in data_path.glob('*.mseed')]

        if reset_training:
            self.training_db_manager.clear_database()

        # if enable_multiprocessing:
        #     num_cores = multiprocessing.cpu_count()
        #     pool = multiprocessing.Pool(num_cores - 10)
        #
        #     list(tqdm(pool.imap(self.process, mseed_files),
        #               total=len(mseed_files)))
        #
        # else:
        for mseed_file in tqdm(mseed_files):
            self.__process__(mseed_file)

    def __process__(self, f):
        cat = read_events(f.with_suffix('.xml'))
        st = read(f)
        st = self.__filter_adequate_trace_length__(st)
        st = st.detrend('linear').detrend('demean')

        noise_counter = 0

        origin = cat[0].preferred_origin()
        if origin is None:
            origin = cat[0].origins[-1]
        if len(st) == 0:
            return
        if event_type_lookup[cat[0].event_type] == 'seismic event':
            if origin.evaluation_status == 'rejected':
                return
            self.__process_seismic_event__(st, cat, f)
        elif event_type_lookup[cat[0].event_type] == 'blast':
            if origin.evaluation_status == 'rejected':
                return
            # if self.__get_magnitude__(cat) < -0.5:
            #     if self.__get_magnitude__(cat) > -3:
            #         self.__process_seismic_event__(st, cat, f)
            self.__process_blast__(st, cat, f)
        else:
            if event_type_lookup[cat[0].event_type] == 'impulsive noise':
                self.__process_noise__(st, cat, f)
            else:
                if np.random.randint(0, 3) == 1:
                    self.__process_noise__(st, cat, f)

    def __process_seismic_event__(self, st, cat, f, snr_threshold=12):
        origin = cat[0].preferred_origin()
        if origin is None:
            origin = cat[0].origins[-1]

        arrival_sites, indices = np.unique([arrival.site for arrival in origin.arrivals],
                                           return_index=True)
        arrival_station = np.array([arrival.station for arrival in
                                    origin.arrivals])[indices]
        arrival_location = np.array([arrival.location for arrival in
                                    origin.arrivals])[indices]

        arrivals = []
        for i, arrival in enumerate(origin.arrivals):
            if i in indices:
                arrivals.append(arrival)

        for station, location, arrival in tqdm(zip(arrival_station, arrival_location,
                                                   arrivals),
                                               total=len(arrival_sites)):

            if arrival.pick.snr is None:
                continue

            if arrival.pick.snr < snr_threshold:
                continue

            st2 = st.select(station=station,
                            location=location).copy()

            if st2 is None:
                continue

            if len(st2) == 0:
                continue

            try:
                end_window = self.__find_end_window__(st2)
            except:
                continue

            if end_window is None:
                continue

            magnitude = self.__get_magnitude__(cat)

            self.create_spectrogram_training(st2, 'seismic event', cat[0].event_type,
                                             f.name, end_window, magnitude,
                                             simulate_magnitude=True)

            # specs = model.EventClassifier2.stream2spectrogram(st2)

    def __find_end_window__(self, st2):
        starttime = st2[0].stats.starttime
        endtime = st2[0].stats.endtime

        if endtime - starttime < self.window_length_seconds[0]:
            return

        max_sample = np.argmax(np.abs(st2.composite()[0].data))
        max_time_from_trace_start = max_sample / st2[0].stats.sampling_rate
        max_time_to_trace_end = endtime - starttime - max_time_from_trace_start

        max_time = st2[0].stats.starttime + max_time_from_trace_start

        if max_time_from_trace_start < 0.2 * self.window_length_seconds[0]:
            start_window = starttime
            end_window = start_window + self.window_length_seconds[0]

        elif max_time_to_trace_end < 0.2 * self.window_length_seconds[0]:
            end_window = endtime
            # start_window = end_window - self.window_length_seconds

        else:
            window = self.window_length_seconds[0] * (0.8 - 0.4 * np.random.rand())

            if max_time_from_trace_start > max_time_to_trace_end:

                start_window = max_time - window
                end_window = start_window + self.window_length_seconds[0]
            else:
                end_window = max_time + window

        return end_window

    def __process_blast__(self, st, cat, f):
        traces = []
        for tr in tqdm(st):
            max_sample = np.argmax(tr.data)
            max_time = tr.stats.starttime + max_sample / tr.stats.sampling_rate

            st2 = Stream(traces=[tr])
            try:
                end_window = self.__find_end_window__(st2)
            except:
                continue

            if end_window is None:
                continue

            magnitude = self.__get_magnitude__(cat)

            self.create_spectrogram_training(Stream(traces=[tr]), 'blast',
                                             cat[0].event_type,
                                             f.name, end_window, magnitude,
                                             simulate_magnitude=False)

    def __process_noise__(self, st, cat, f):

        # st2 = self.__select_trace_based_on_amplitude__(st)

        st2 = self.__select_based_on_distance__(st, cat)

        for tr in tqdm(st2):
            starttime = tr.stats.starttime
            endtime = tr.stats.endtime

            min_end_window = starttime + self.window_length_seconds[0]

            available_end_window = endtime - min_end_window

            end_window = min_end_window + np.random.rand() * available_end_window

            magnitude = self.__get_magnitude__(cat)

            self.create_spectrogram_training(Stream(traces=[tr]), 'noise',
                                             cat[0].event_type,
                                             f.name, end_window, magnitude)

    @staticmethod
    def __select_trace_based_on_amplitude__(st, n_traces=50):
        maxes = []
        for tr in st:
            maxes.append(tr.data.max() - tr.data.min())

        indices = np.argsort(maxes)

        traces = []
        st2 = st.copy()
        for i in indices[:n_traces]:
            traces.append(st2[i])

        return Stream(traces=traces)

    def __select_based_on_distance__(self, st, cat, n_sites=20):
        origin = cat[0].preferred_origin()
        if origin is None:
            origin = cat[0].origins[-1]

        loc = origin.loc

        distances = []
        sites = self.inventory.sites
        for site in sites:
            distances.append(np.linalg.norm(site.loc - origin.loc))

        indices = np.argsort(distances)
        selected_sites = sites[indices[0:n_sites]]

        traces = []
        for site in selected_sites:
            station = site.station_code
            location = site.location_code
            st2 = st.copy().select(station=station, location=location)

            for tr in st2:
                traces.append(tr)

        return Stream(traces=traces)

    @staticmethod
    def __get_magnitude__(cat):
        magnitude = cat[0].preferred_magnitude()
        if magnitude is None:
            if len(cat[0].magnitudes) > 0:
                magnitude = cat[0].magnitudes[-1]
            else:
                return -999
        if magnitude.mag is None:
            return -999
        return magnitude.mag

    def __filter_adequate_trace_length__(self, st):
        traces = []
        for tr in st:
            trace_length = tr.stats.endtime - tr.stats.starttime
            if trace_length < self.window_length_seconds[0]:
                continue
            if np.any(np.isnan(tr.data)) or np.any(np.isinf(tr.data)):
                continue
            traces.append(tr)
        return Stream(traces=traces)

    def predict(self, st: uquake.core.stream.Stream, cut_from_start=True,
                event_location=None):
        """
        :param st: the waveforms
        :type st: uquake.core.stream.Stream
        :return:
        """

        if cut_from_start:
            cut_from = 'start'
        else:
            cut_from = 'end'

        specs = self.spectrogram_stream(st, trim_from_start=cut_from_start)
        specs = specs.to(self.event_classifier.device)

        return ClassifierResult(self.event_classifier.model(specs).cpu(),
                                self.label_mapping, st, specs, self.inventory.copy(),
                                event_location=event_location)

        # # st2 = self.__filter_adequate_trace_length__(st.copy())
        # # set_trace()
        #
        # self.event_classifier.model.eval()
        # images = []
        # for tr in st.copy():
        #     tr.detrend('demean').detrend('linear')
        #     trace_length = tr.stats.endtime - tr.stats.starttime
        #     if trace_length > self.window_length_seconds[0]:
        #         logger.warning(f'The traces {tr.stats.site} length ({trace_length} '
        #                        f'is longer than {self.window_length_seconds[0]:0.2f} '
        #                        f'and '
        #                        f'will be cut from the {cut_from} of the trace')
        #         # tr2 = tr.trim(starttime=tr.stats.starttime, endtime=tr.stats.starttime +
        #         #               self.window_length_seconds[0], pad=True, fill_value=0)
        #
        #     # elif trace_length < self.window_length_seconds[0]:
        #     #     logger.warning(f'The traces {tr.stats.site} length ({trace_length} '
        #     #                    f'is shorter than {self.window_length_seconds[0]:0.2f} '
        #     #                    f'and '
        #     #                    f'will be padded to fit the required length. This may '
        #     #                    f'result is a reduced prediction accuracy')
        #     #
        #     #     tr2 = tr.trim(starttime=tr.stats.starttime, endtime=tr.stats.starttime +
        #     #                   self.window_length_seconds[0], pad=True, fill_value=0)
        #
        #     try:
        #         # specs = self.trace2spectrograms(tr2.copy())[0][0]
        #
        #     except Exception as e:
        #         logger.error(e)
        #         continue
        #     set_trace()
        #     image = dataset.SpectrogramDataset2.transform(specs)
        #     image_tensor = image.unsqueeze(0)
        #
        #     images.append(image_tensor)
        #
        # merged_images = torch.cat(images, dim=0)
        #
        # merged_images = merged_images.to(self.event_classifier.device)
        #
        # return ClassifierResult(self.event_classifier.model(merged_images).cpu(),
        #                         self.label_mapping, st, images, self.inventory.copy(),
        #                         event_location=event_location)




