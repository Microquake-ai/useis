from torchvision import models
from torchvision.models import resnet34
from .dataset import (Dataset, PickingDataset, ClassifierDataset1D,
                      spectrogram)
from torch.utils.data import DataLoader
import torch
from torch import nn
import numpy as np
from loguru import logger
from tqdm import tqdm
import pickle
from sklearn.metrics import confusion_matrix
# from plotcm import plot_confusion_matrix
from uquake.core import Stream, Trace, UTCDateTime
# from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from PIL import Image, ImageOps
from torch import nn
from .resnet1d import ResNet1D
from importlib import reload
from ipdb import set_trace
# import urllib.request
import requests
from io import BytesIO
from uquake.core.util.requests import download_file_from_url

from .params import *


# class SeismicResnet34(type(resnet34())):
#     def __init__(self, num_input_channels=3, n_out_features=3):
#         super(SeismicResnet34, self).__init__()
#         self.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2,
#                                padding=3, bias=False)
#
#         self.fc = nn.Linear(self.fc.in_features, n_out_features)


class EventClassifier(object):

    model_url = \
            'https://www.dropbox.com/s/5d76v8hfi3dwsm0/classification_model.useis?dl=1'
    n_out_features = 3
    label_mapping = {
        'noise': np.array([1., 0., 0.]),
        'blast': np.array([0., 1., 0.]),
        'seismic event': np.array([0., 0., 1.])
    }

    def __init__(self, gpu: bool = True, learning_rate: float = 0.001, model_id=None,
                 model=models.resnet34(), weight_decay: float = 0.001):

        # define the model
        self.model = model
        num_input_channels = 3
        self.num_imput_channels = num_input_channels
        self.model.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2,
                                     padding=3, bias=False)

        self.model.fc = nn.Linear(self.model.fc.in_features, self.n_out_features)

        self.model_id = model_id

        self.n_features = self.n_out_features
        self.num_classes = self.n_out_features
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Replace the last fully-connected layer to output the desired number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, self.n_out_features)

        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate,
                                          weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.gpu = gpu

        self.device = self.select_device(gpu=gpu)

        self.model = self.model.to(self.device)
        self.accuracies = []
        self.losses = []

        self.iteration = 0
        # Download the model from Dropbox

    @classmethod
    def from_pretrained_model(cls, model, gpu: bool = True):
        cls_tmp = cls(gpu=gpu, model=model)
        # cls_tmp.model = model.to(cls_tmp.device)
        return cls_tmp

    @staticmethod
    def select_device(gpu=True):
        if gpu:
            device = torch.device("cuda:0" if
                                  torch.cuda.is_available() else "cpu")
            if device == "cpu":
                logger.warning('GPU is not available, the CPU will be used')
            else:
                logger.info('GPU will be used')
        else:
            device = torch.device('cpu')
            logger.info('The CPU will be used')

        return device


    @classmethod
    def from_pretrained_model_file(cls, file_name, gpu: bool = True):
        # with open(path, 'rb') as input_file:
        model = torch.load(file_name, map_location=cls.select_device(gpu=gpu))
        return cls.from_pretrained_model(model, gpu=gpu)

    # @classmethod
    # def load_model(cls, file_name):
    #     return pickle.load(open('filename', 'rb'))

    def train(self, dataset: Dataset, batch_size: int):

        self.model.train()

        self.accuracies = []
        self.losses = []

        training_loader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=True)

        for inputs, targets in tqdm(training_loader):
            self.iterate(inputs, targets)
            torch.cuda.empty_cache()

    def forward(self, x):
        return self.model.forward(x)

    def iterate(self, inputs, targets):

        self.optimizer.zero_grad()

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # with torch.no_grad():
        predictions = self.model(inputs)

        loss = self.criterion(predictions, targets)
        # loss.requires_grad = True
        loss.backward()
        self.optimizer.step()

        target_indices = torch.argmax(targets, dim=1)
        predicted_indices = torch.argmax(predictions, dim=1)
        loss = self.criterion(predictions, target_indices)
        accuracy = (target_indices == predicted_indices).sum() / len(target_indices)
        self.accuracies.append(accuracy.item())
        self.losses.append(loss.item())
        self.iteration += 1

    def validate(self, dataset: Dataset, batch_size: int):
        self.model.eval()
        training_loader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=True)
        self.validation_targets = np.zeros(len(dataset))
        self.validation_predictions = np.zeros(len(dataset))
        accuracy = []
        loss = []
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(tqdm(training_loader)):
                # inputs = inputs.view(inputs.size()[0], -1, inputs.size()[1],
                #                      inputs.size()[2])
                inputs = inputs.to(self.device)

                start = batch_size * i
                end = start + batch_size
                self.validation_targets[start:end] = torch.argmax(targets).cpu().numpy()
                targets = targets.to(self.device)

                # with torch.no_grad() pytorch does not compute the gradient,
                # the gradient takes a significant amount of memory on the gpu.
                # using torch.no_grad() consequently allows significantly
                # increasing the batch size
                # with torch.no_grad():
                outputs = self.model(inputs)

                self.validation_predictions[start:end] = torch.argmax(
                    outputs).cpu().numpy()

                target_indices = torch.argmax(targets, dim=1)
                predicted_indices = torch.argmax(outputs, dim=1)
                tmp = (target_indices == predicted_indices).sum() / len(target_indices)
                accuracy.append(float(tmp.cpu().numpy()))
                loss.append(float(self.criterion(outputs, target_indices).cpu().numpy()))
        self.model.train()
        return accuracy, loss

    def display_memory(self):
        torch.cuda.empty_cache()
        memory = torch.cuda.memory_allocated(self.device)
        logger.info("{:.3f} GB".format(memory / 1024 ** 3))

    # @property
    # def model_state(self):
    #     self.model.to('cpu')
    #     out_dict = {'model_state': self.model.state_dict(),
    #                 'out_features': self.n_features,
    #                 'label_map': self.label_mapping,
    #                 'learning_rate': self.learning_rate,
    #                 'model_id': self.model_id}
    #     return out_dict

    def save(self, file_name):
        return torch.save(self.model.state_dict(), file_name)

    def write(self, file_name):
        return self.save(file_name)

    @classmethod
    def from_model_state(cls, state_dict, gpu=True):
        ec = cls(gpu=gpu)
        ec.model.load_state_dict(state_dict)
        ec.model.eval()
        ec.model.to(cls.select_device(gpu=gpu))
        return ec

    @classmethod
    def read(cls, file_name, gpu=True):
        device = cls.select_device(gpu=gpu)
        state_dict = torch.load(file_name, map_location=device)
        # model_state = pickle.load(open(file_name, 'rb'))
        return cls.from_model_state(state_dict, gpu=gpu)

    @classmethod
    def load(cls, gpu=True):
        """
        load the most recent model
        """

        model_data = download_file_from_url(cls.model_url)

        # Load the model from the BytesIO object

        device = cls.select_device(gpu=gpu)

        state_dict = torch.load(model_data, map_location=device)
        return cls.from_model_state(state_dict, gpu=gpu)
        # model = MyModel(...)
        # model.model.load_state_dict(state_dict['model_state'])
        # model.n_features = state_dict['out_features']
        # model.label_mapping = state_dict['label_map']
        # model.learning_rate = state_dict['learning_rate']
        # model.model_id = state_dict['model_id']

    @staticmethod
    def measure_accuracy(targets, predictions):
        max_index = predictions.max(dim=1)[1]
        return (max_index == targets).sum() / len(targets)


class EventClassifier2(EventClassifier):
    model_url = 'https://www.dropbox.com/s/klk2nfalqt8ugjj/1s_model_2023_04_30.pt?dl=1'

    def __init__(self, gpu: bool = True,
                 learning_rate: float = 0.001, model=models.resnet34(), model_id=None,
                 weight_decay: float = 0.001, dropout_prob: float = 0.3):
        super().__init__(gpu=gpu,
                         learning_rate=learning_rate, model_id=model_id, model=model,
                         weight_decay=weight_decay)
        self.num_input_channels = 1

        self.model.conv1 = nn.Conv2d(self.num_input_channels, 64, kernel_size=7,
                                     stride=2, padding=3, bias=False)

        self.model.to(self.device)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.resnet34.conv1(x)
        x = self.resnet34.bn1(x)
        x = self.resnet34.relu(x)
        x = self.resnet34.maxpool(x)

        # Stage 1
        x = self.resnet34.layer1(x)

        # Stage 2
        x = self.resnet34.layer2(x)

        # Stage 3
        x = self.resnet34.layer3(x)

        # Stage 4
        x = self.resnet34.layer4(x)

        # Apply dropout to the output of the last convolutional layer
        x = self.dropout(x)

        x = self.resnet34.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.resnet34.fc(x)
        return x

    @staticmethod
    def trace2spectrogram(trace: Trace):
        return generate_spectrogram(Stream(traces=[trace]))

    @staticmethod
    def stream2spectrogram(stream: Stream):
        return generate_spectrogram(stream)

    @property
    def model_state(self):
        out_dict = {'model_state': self.model.state_dict(),
                    'out_features': self.n_features,
                    'label_map': self.label_mapping,
                    'learning_rate': self.learning_rate,
                    'model_id': self.model_id}
        return out_dict

    # def save(self, file_name):
    #     return pickle.dump(self.model_state, open(file_name, 'wb'))
    #
    # def write(self, file_name):
    #     return self.save(file_name)
    #
    # @classmethod
    # def from_model_state(cls, model_state, gpu=True):
    #     ec = cls(model_state['out_features'], model_state['label_map'],
    #              gpu=gpu, learning_rate=model_state['learning_rate'],
    #              model_id=model_state['model_id'])
    #     ec.model.load_state_dict(model_state['model_state'])
    #     ec.model.eval()
    #     return ec

    # @classmethod
    # def read(cls, file_name, gpu=True):
    #     model_state = pickle.load(open(file_name, 'rb'))
    #     return cls.from_model_state(model_state, gpu=gpu)


def read_classifier(file_name):
    return EventClassifier.read(file_name)


def generate_spectrogram(stream: Stream):

    specs = []
    trs = []
    for tr in stream.copy():
        # check if tr contains NaN
        if np.any(np.isnan(tr.data)):
            continue
        trs.append(tr.copy())

    st2 = Stream(traces=trs)
    st2 = st2.detrend('demean').detrend('linear')

    st2 = st2.taper(max_percentage=1, max_length=1e-2)

    for tr in st2:
        spec = spectrogram(tr)
        spec /= np.max(spec)
        spec = spec / np.max(spec) * 255
        spec = Image.fromarray(np.array(spec.tolist()).astype(np.uint8))
        specs.append(spec)

    return specs


class EventClassifier1D(EventClassifier):
    def __init__(self, n_classes: int, in_channels: int = 1,
                 base_filters: int = 16, kernel_size: int = 7,
                 stride: int = 3, groups: int = 1, n_block: int = 8,
                 learning_rate=1e-3, gpu: bool = True, weight_decay: float = 0.001):

        if gpu:
            device = torch.device("cuda:0" if
                                  torch.cuda.is_available() else "cpu")
            if device == "cpu":
                logger.warning('GPU is not available, the CPU will be used')
            else:
                logger.info('GPU will be used')
        else:
            device = 'cpu'
            logger.info('The CPU will be used')

        self.device = device

        self.model = ResNet1D(in_channels, base_filters, kernel_size,
                              stride, groups, n_block,
                              n_classes).to(self.device)

        # self.model = Conv1DModel()

        self.in_channels = in_channels
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_classes = n_classes
        self.groups = groups
        self.n_block = n_block
        self.gpu = gpu

        self.model = ResNet1D(in_channels, base_filters, kernel_size,
                              stride, groups, n_block,
                              n_classes).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate,
                                          weight_decay=weight_decay)
        # self.optimizer = torch.optim.SGD(self.model.parameters(),
        #                                  lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # self.display_memory()
        self.losses = []

        self.validation_predictions = None
        self.validation_targets = None

    def train(self, dataset: ClassifierDataset1D, batch_size: int):

        training_loader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=True)

        for inputs, targets in tqdm(training_loader):
            inputs = inputs.view(inputs.size()[0], -1, inputs.size()[1])
            self.iterate(inputs, targets)
            torch.cuda.empty_cache()

    def iterate(self, inputs, targets):

        self.optimizer.zero_grad()

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        predictions = self.model(inputs)

        # loss = self.criterion(predictions, targets.view(len(targets), -1))
        loss = self.criterion(predictions, targets)
        # print(loss.item())
        loss.backward()
        self.optimizer.step()

        # print(loss.item())
        self.losses.append(loss.cpu().item())
        self.iteration += 1

    def validate(self, dataset: ClassifierDataset1D, batch_size: int):
        self.model.eval()
        training_loader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=True)
        self.validation_targets = np.zeros(len(dataset))
        self.validation_predictions = np.zeros(len(dataset))
        accuracy = 0
        for i, (inputs, targets) in enumerate(tqdm(training_loader)):
            inputs = inputs.view(inputs.size()[0], -1, inputs.size()[1])
            inputs = inputs.to(self.device)

            start = batch_size * i
            end = start + batch_size
            self.validation_targets[start:end] = targets
            targets = targets.to(self.device)

            # with torch.no_grad() pytorch does not compute the gradient,
            # the gradient takes a significant amount of memory on the gpu.
            # using torch.no_grad() consequently allows significantly
            # increasing the batch size
            with torch.no_grad():
                outputs = self.model(inputs)

            self.validation_predictions[start:end] = outputs.max(dim=1)[1].cpu(
            ).detach().numpy()

            accuracy += self.measure_accuracy(targets, outputs)
        self.model.train()
        return accuracy.cpu().detach().numpy() / (i + 1)


class Conv1DModel(nn.Module):
    def __init__(self):
        super(Conv1DModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, 15),
            nn.ReLU(),
            nn.Conv1d(64, 64, 15),
            nn.ReLU(),
            nn.Conv1d(64, 64, 15),
            nn.ReLU(),
            nn.Conv1d(64, 128, 15),
            nn.ReLU(),
            nn.Conv1d(128, 128, 15),
            nn.ReLU(),
            nn.Conv1d(128, 128, 15),
            nn.ReLU()
        )

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=128 * 172, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1))

    def forward(self, X):
        out = self.conv_layers(X)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)
        # input(out)
        return out


# class AIPicker(EventClassifier):
class AIPicker(object):
    def __init__(self, in_channels: int = 1, base_filters: int = 64,
                 kernel_size: int = 15, stride: int = 3, n_classes: int = 1,
                 groups: int = 1, n_block: int = 16, learning_rate=1e-5,
                 gpu: bool = True, n_sample: int = 256,
                 sampling_rate: int=6000, to_device=True):
        self.in_channels = in_channels
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_classes = n_classes
        self.groups = groups
        self.n_block = n_block
        self.gpu = gpu
        self.n_sample = n_sample
        self.sampling_rate = sampling_rate

        if gpu:
            device = torch.device("cuda:0" if
                                  torch.cuda.is_available() else "cpu")
            if device == "cpu":
                logger.warning('GPU is not available, the CPU will be used')
            else:
                logger.info('GPU will be used')
        else:
            device = 'cpu'
            logger.info('The CPU will be used')

        self.device = device

        self.model = self.init_model()

        if to_device:
            self.model.to(self.device)
        # self.model = ResNet1D(self.in_channels, self.base_filters, self.kernel_size,
        #                       self.stride, self.groups, self.n_block,
        #                       n_classes).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.losses = []

        self.validation_predictions = None
        self.validation_targets = None

    def init_model(self):
        return ResNet1D(self.in_channels, self.base_filters, self.kernel_size,
                        self.stride, self.groups, self.n_block,
                        self.n_classes)

    def train(self, dataset: PickingDataset, batch_size: int):

        self.model.train()

        training_loader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=True)

        for inputs, targets in tqdm(training_loader):
            inputs = inputs.view(inputs.size()[0], -1, inputs.size()[1])
            self.iterate(inputs, targets)
            torch.cuda.empty_cache()

    def iterate(self, inputs, targets):

        self.optimizer.zero_grad()

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        predictions = self.model(inputs)

        loss = self.criterion(predictions, targets.view(len(targets), -1))
        # print(loss.item())
        loss.backward()
        self.optimizer.step()

        # print(loss.item())
        self.losses.append(loss.cpu().item())

    def predict(self, inputs):
        if isinstance(inputs, np.ndarray):
            if len(inputs.shape) == 1:
                dim1 = 1
                dim2 = inputs.shape[0]
            else:
                dim1 = inputs.shape[0]
                dim2 = inputs.shape[1]

            inputs = torch.from_numpy(inputs)
        elif not isinstance(inputs, torch.Tensor):
            raise TypeError

        inputs = inputs.view(dim1, -1, dim2).to(self.device)
        with torch.no_grad():
            outputs_tensor = self.model(inputs)

        outputs = [out.item() for out in outputs_tensor.cpu()]
        return outputs

    def predict_trace(self, trace: Trace, pick_time: UTCDateTime, phase: str):
        # import matplotlib.pyplot as plt

        trace = trace.resample(sampling_rate=int(self.sampling_rate))

        n_sample = int((pick_time - trace.stats.starttime)
                       * trace.stats.sampling_rate)

        if phase == 'P':
            n_sample_start = int(n_sample - self.n_sample * 0.5)
        else:
            n_sample_start = int(n_sample - self.n_sample * 0.5)

        n_sample_end = int(n_sample_start + self.n_sample)

        data = trace.data[n_sample_start: n_sample_end]
        data -= np.mean(data)
        data /= np.max(np.abs(data))

        predicted_arrival = self.predict(data)[0]

        # import matplotlib.pyplot as plt
        # plt.clf()
        # plt.plot(data)
        # plt.axvline(predicted_arrival)
        # plt.show()
        # input(predicted_arrival)

        # return n_sample_start + predicted_arrival * self.n_sample

        new_pick_time = trace.stats.starttime + \
                        (n_sample_start + predicted_arrival *
                        self.n_sample) / self.sampling_rate

        return new_pick_time

    def predict_stream(self, st: Stream, pick_times):
        pass

    def validate(self, dataset: PickingDataset, batch_size):
        predictions = []
        training_loader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=True)

        losses = []
        for inputs, targets in tqdm(training_loader):
            inputs = inputs.view(inputs.size()[0], -1, inputs.size()[1])

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            with torch.no_grad():
                ps = self.model(inputs)

            for p in ps.cpu():
                predictions.append(p.item())

            loss = self.criterion(ps, targets.view(len(targets), -1))
            # print(loss.cpu().item())
            losses.append(loss.cpu().item())
        return predictions, np.mean(losses), np.std(losses)

    def save(self, file_name):
        with(open(file_name, 'wb')) as f_out:
            pickle.dump(self, f_out)
        # torch.save(self.model.state_dict(), file_name)

    def write(self, file_name):
        self.save(file_name)

    @classmethod
    def read(cls, file_name):
        with open(file_name, 'rb') as f_in:
            return pickle.load(f_in)

    @classmethod
    def from_model(cls, file_name, gpu: bool = False):
        ai_picker = cls(gpu=gpu)
        with(open(file_name, 'rb')) as f_in:
            picker = pickle.load(f_in)
            ai_picker.model = picker.model
            ai_picker.model.to(ai_picker.device)
            ai_picker.model = ai_picker.model.double()
            ai_picker.model = ai_picker.model.eval()
        # picker = cls()
        # picker.model.load_state_dict(torch.load(filename))
        # picker.model.eval()
        # picker.model.to(self.device)
        #
        # picker.in_channels = in_channels
        # self.base_filters = base_filters
        # self.kernel_size = kernel_size
        # self.stride = stride
        # self.n_classes = n_classes
        # self.groups = groups
        # self.n_block = n_block
        # self.gpu = gpu
        # self.n_sample = n_sample
        # self.sampling_rate = sampling_rate

        # if device == 'gpu':
        #     device = torch.device("cuda:0" if
        #                           torch.cuda.is_available() else "cpu")
        #     if device == "cpu":
        #         logger.warning('GPU is not available, '
        #                        'the CPU will be used')
        #     else:
        #         logger.info('GPU will be used')
        # else:
        #     device = 'cpu'
        #     logger.info('The CPU will be used')

        return ai_picker


def read_picker(file_name):
    return AIPicker.read(file_name)



