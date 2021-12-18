from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path
from os.path import sep
from loguru import logger
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from uquake.core.trace import Trace
from abc import ABC
import pickle


# class Label():
#     def __init__(self, labels):
#         self.full_labels = list(labels)
#         self.labels = range(0, len(labels))
#
#         self.unique_labels = np.unique(labels)
#         self.unique_numerical_labels = np.range(len(self.unique_labels))
#
#         self.labels_dict = {}
#         for full_label, label  in zip(labels, self.unique_labels):
#             self.labels_dict[full_label] = label
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, item):
#         label = self.full_labels[item]
#         return self.labels_dict[label]
#
#     def get_full_label(self, item):

def get_file_list(input_directory, suffix, extension='png'):
    path = os.path.join(input_directory, '*', f'*{suffix}.{extension}')
    return glob(path)


def split_dataset(path, split=0.8, seed=None, max_item_per_category=100000):
    """

    :param path: list of image files
    :param split: set_1 fraction (between 0 and 1)
    :param seed: seed for the random number generator to provide
    reproducibility (Default None)
    :param max_item_per_category: maximum number of item per category
    :return: set_1, set_2
    """

    np.random.seed(seed)
    choices = np.random.choice([False, True], size=len(file_list),
                               p=[1-split, split])
    not_choices = [not choice for choice in choices]
    set_1 = np.array(file_list)[choices]
    set_2 = np.array(file_list)[not_choices]

    return set_1, set_2


def split_filename_label(file_list):
    file_list = [os.path.split(f)[-1] for f in tmp]
    labels = [os.path.split(f)[0].split('/')[-1] for f in tmp]


class FileList(object):
    def __init__(self, path, unused_category=['unknown'], seed=None,
                 extension='.jpg'):

        self.path = Path(path)
        dir_list = self.path.glob('*')

        self.category_list = [str(dr).split(sep)[-1] for dr in dir_list
                              if str(dr).split(sep)[-1] not in unused_category]
        self.labels = np.arange(len(self.category_list))

        self.labels_dict = {}
        np.random.seed(seed)
        self.cat_file_list = {}
        for label, category in zip(self.labels, self.category_list):
            logger.info(f'processing {category}')
            self.labels_dict[category] = label

            self.cat_file_list[category] = [fle for fle in
                                            (self.path /
                                             category).glob(f'*.{extension}')]

        self.classes = self.category_list

    def select(self, number):
        return SpectrogramDataset.from_file_list(self,
                                                 max_number_image_per_category
                                                 =number)

    def select1d(self, number):
        return ClassifierDataset1D.from_file_list(self,
                max_number_signal_per_category=number)


class SpectrogramDataset(Dataset):

    def __init__(self, file_dict: dict,
                 max_number_image_per_category: int = 1e5,
                 seed: int = None):

        categories = [key for key in file_dict.keys()]
        self.category_list = categories
        self.labels = np.arange(len(categories))

        np.random.seed(seed)
        self.labels_dict = {}
        self.file_list = []
        self.label_list = []
        for category, label in zip(categories, self.labels):
            self.labels_dict[category] = label
            nb_files = int(max_number_image_per_category)
            if len(file_dict[category]) < nb_files:
                nb_files = int(len(file_dict[category]))
            for f in np.random.choice(file_dict[category],
                                      size=nb_files,
                                      replace=False):
                self.file_list.append(f)
                self.label_list.append(label)

        self.file_list = np.array(self.file_list)
        self.label_list = np.array(self.label_list)

    @classmethod
    def from_file_list(cls, file_list: FileList,
                       max_number_image_per_category=1e5, seed=None):
        return cls(file_list.cat_file_list,
                   max_number_image_per_category=max_number_image_per_category,
                   seed=seed)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # label = self.df_data.iloc[idx]['label']
        # category = self.category_list[idx]
        label = self.label_list[idx]

        image = torch.from_numpy((np.array(Image.open(
            self.file_list[idx])) / 255).astype(np.float32))

        # return {'data': image, 'label': label}
        return image, label

    @property
    def shape(self):
        px_x, px_y = self[0][0].shape
        return len(self.label_list), px_x, px_y

    @property
    def nb_pixel(self):
        return np.prod(self[0][0].shape)

    @property
    def categories(self):
        return self.labels_dict

    @property
    def classes(self):
        return self.categories

    @property
    def nb_categories(self):
        return len(self.category_list)


sampling_rate = 6000
# num_threads = int(np.ceil(cpu_count() - 10))
num_threads = 10
replication_level = 5
snr_threshold = 10
sequence_length_second = 2
perturbation_range_second = 1
image_width = 128
image_height = 128
buffer_image_fraction = 0.05

buffer_image_sample = int(image_width * buffer_image_fraction)

hop_length = int(sequence_length_second * sampling_rate //
                 (image_width + 2 * buffer_image_sample))


def spectrogram(trace: Trace):

    trace.resample(sampling_rate)

    mel_spec = MelSpectrogram(sample_rate=sampling_rate,
                              n_mels=image_height,
                              hop_length=hop_length,
                              power=1,
                              pad_mode='reflect',
                              normalized=True)

    amplitude_to_db = AmplitudeToDB()

    # trace = trace.detrend('linear')
    # trace = trace.detrend('demean')
    trace.data = trace.data - np.mean(trace.data)
    trace = trace.taper(max_length=0.01, max_percentage=0.05)
    trace = trace.trim(starttime=trace.stats.starttime,
                       endtime=trace.stats.starttime + sequence_length_second,
                       pad=True, fill_value=0)
    data = trace.data

    torch_data = torch.tensor(data).type(torch.float32)

    spec = (mel_spec(torch_data))
    spec_db = amplitude_to_db(spec.abs() + 1e-3)
    spec_db = (spec_db - spec_db.min()).numpy()
    # spec_db = (spec_db / spec_db.max()).type(torch.float32)
    return spec_db


class ClassifierDataset1D(Dataset):

    def __init__(self, file_dict: dict,
                 max_number_signal_per_category: int = 1e5,
                 seed: int = None):

        categories = [key for key in file_dict.keys()]
        self.category_list = categories
        self.labels = np.arange(len(categories))

        np.random.seed(seed)
        self.labels_dict = {}
        self.file_list = []
        self.label_list = []
        for category, label in zip(categories, self.labels):
            self.labels_dict[category] = label
            nb_files = int(max_number_signal_per_category)
            if len(file_dict[category]) < nb_files:
                nb_files = int(len(file_dict[category]))
            for f in np.random.choice(file_dict[category],
                                      size=nb_files,
                                      replace=False):
                self.file_list.append(f)
                self.label_list.append(label)

        self.file_list = np.array(self.file_list)
        self.label_list = np.array(self.label_list)

    @classmethod
    def from_file_list(cls, file_list: FileList,
                       max_number_signal_per_category=1e5, seed=None):
        return cls(file_list.cat_file_list,
                   max_number_signal_per_category=
                   max_number_signal_per_category,
                   seed=seed)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with open(self.file_list[idx], 'rb') as f_in:
            data_dict = pickle.load(f_in)

        # label = self.df_data.iloc[idx]['label']
        # category = self.category_list[idx]
        label = self.label_list[idx]

        data = data_dict['data']

        if idx != 0:
            data_out = np.zeros(self.shape[1])
            sample_diff = len(data) - self.shape[1]
            if sample_diff > 0:
                data_out = data[0:self.shape[1]]
            else:
                data_out[0: len(data)] = data
        else:
            data_out = data

        data_out -= np.mean(data_out)
        data_out /= np.max(np.abs(data_out))

        data_out = torch.from_numpy((np.array(data_out).astype(np.float32)))

        return data_out, label

    @property
    def shape(self):
        len_data = len(self[0][0])
        return len(self.label_list), len_data

    @property
    def categories(self):
        return self.labels_dict

    @property
    def classes(self):
        return self.categories

    @property
    def nb_categories(self):
        return len(self.category_list)


class PickingDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with open(self.file_list[idx], 'rb') as f_in:
            data = pickle.load(f_in)

        signal = data['data'].astype(np.float32)
        # signal += np.min(signal)
        # signal = np.abs(signal) / np.abs(signal)

        # ensuring the signal is between -1 and 1
        signal = signal - np.mean(signal)
        signal = signal / np.max(np.abs(signal))

        signal[np.isnan(signal)] = 0

        target = data['pick']
        if np.isnan(target):
            targer = 0

        return signal.astype(np.float32), np.float32(target)

    @property
    def shape(self):
        len_signal = self[0][0].shape
        return 1, len_signal[0]

    def split(self, split_fraction: float = 0.8, random_seed: int = None):
        np.random.seed(random_seed)
        choices = np.random.choice([False, True], size=len(self),
                                   p=[1 - split_fraction, split_fraction])
        not_choices = np.invert(choices)
        files_set_a = np.array(self.file_list)[choices]
        files_set_b = np.array(self.file_list)[not_choices]

        set_a = PickingDataset(files_set_a)
        set_b = PickingDataset(files_set_b)

        return set_a, set_b







