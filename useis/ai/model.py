from torchvision import models
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
# from .dataset as dataset
# reload(dataset)


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


class EventClassifier(object):
    def __init__(self, n_features: int, gpu: bool = True):

        # define the model
        self.model = models.resnet34(pretrained=False)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                     bias=False)

        self.n_features = n_features

        self.model.fc = nn.Linear(self.model.fc.in_features, n_features)

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        self.gpu = gpu

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

        self.model = self.model.to(self.device)
        self.display_memory()
        self.accuracies = []
        self.losses = []

        self.validation_predictions = None
        self.validation_targets = None

    @classmethod
    def from_pretrained_model(cls, model, gpu: bool = True):
        n_features = model.fc.out_features
        cls_tmp = cls(n_features, gpu=gpu)
        cls_tmp.model = model.to(cls_tmp.device)

        return cls_tmp

    @classmethod
    def from_pretrained_model_file(cls, path, gpu: bool = True):
        with open(path, 'rb') as input_file:
            model = pickle.load(input_file)

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
            inputs = inputs.view(inputs.size()[0], -1, inputs.size()[1],
                                 inputs.size()[2])
            self.iterate(inputs, targets)
            torch.cuda.empty_cache()

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

        self.accuracies.append(self.measure_accuracy(targets,
                                                     predictions).item())
        self.losses.append(loss.item())

    def validate(self, dataset: Dataset, batch_size: int):
        self.model.eval()
        training_loader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=True)
        self.validation_targets = np.zeros(len(dataset))
        self.validation_predictions = np.zeros(len(dataset))
        accuracy = 0
        for i, (inputs, targets) in enumerate(tqdm(training_loader)):
            inputs = inputs.view(inputs.size()[0], -1, inputs.size()[1],
                                 inputs.size()[2])
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

    def display_memory(self):
        torch.cuda.empty_cache()
        memory = torch.cuda.memory_allocated(self.device)
        logger.info("{:.3f} GB".format(memory / 1024 ** 3))

    def save(self, file_name):
        return pickle.dump(self, open(file_name, 'wb'))

    def write(self, file_name):
        return self.save(file_name)

    @classmethod
    def read(cls, file_name):
        with open(file_name, 'rb') as f_in:
            return pickle.load(f_in)

    @staticmethod
    def measure_accuracy(targets, predictions):
        max_index = predictions.max(dim=1)[1]
        return (max_index == targets).sum() / len(targets)

    def predict(self, stream: Stream):
        """

        :param stream: A uquake.core.stream.Stream object containing the
        waveforms
        :return:
        """
        specs = generate_spectrogram(stream)
        return self.predict_spectrogram(specs)

    def predict_spectrogram(self, specs):
        self.model.eval()
        if isinstance(specs, list):
            specs = np.array(specs)
        if len(specs.shape) == 2:
            specs = torch.from_numpy(specs).view(1, 1, specs.shape[0],
                                                 specs.shape[1])
        else:
            specs = torch.from_numpy(specs).view(specs.shape[0], 1,
                                                 specs.shape[1],
                                                 specs.shape[2])

        specs = specs.to(self.device)
        # self.model.train()
        with torch.no_grad():
            predictions = (self.model(specs).argmax(axis=1).cpu(),
                           self.model(specs).cpu())

        self.model.train()
        return predictions


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
    st2 = st2.trim(endtime=tr.stats.starttime + 2, pad=True,
                 fill_value=0)
    st2 = st2.detrend('demean').detrend('linear')

    st2 = st2.taper(max_percentage=1, max_length=1e-2)

    for tr in st2:
        spec = spectrogram(tr)
        spec /= np.max(spec)

        specs.append(spec)

    return specs


class EventClassifier1D(EventClassifier):
    def __init__(self, n_classes: int, in_channels: int = 1,
                 base_filters: int = 16, kernel_size: int = 7,
                 stride: int = 3, groups: int = 1, n_block: int = 8,
                 learning_rate=1e-3, gpu: bool = True):

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
                                          lr=learning_rate)
        # self.optimizer = torch.optim.SGD(self.model.parameters(),
        #                                  lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        self.display_memory()
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


class AIPicker(EventClassifier):
    def __init__(self, in_channels: int = 1, base_filters: int = 64,
                 kernel_size: int = 15, stride: int = 3, n_classes: int = 1,
                 groups: int = 1, n_block: int = 16, learning_rate=1e-3,
                 gpu: bool = True, n_sample: int = 256):
        self.in_channels = in_channels
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_classes = n_classes
        self.groups = groups
        self.n_block = n_block
        self.gpu = gpu
        self.n_sample = n_sample

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

        # self.model = PickerModel().to(device)

        # self.model = nn.Sequential(
        #     nn.Conv1d(1, 64, 16),
        #     nn.Conv1d(64, 64, 16),
        #     nn.Conv1d(64, 64, 16),
        #     nn.Conv1d(64, 128, 16),
        #     nn.Conv1d(128, 128, 16),
        #     nn.Conv1d(128, 128, 16),
        #     nn.Linear(in_features=12 * 15, out_features=120),
        #     nn.Linear(in_features=120, out_features=60),
        #     nn.Linear(in_features=60, out_features=30),
        #     nn.Linear(in_features=30, out_features=1)
        # ).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate)
        # self.optimizer = torch.optim.SGD(self.model.parameters(),
        #                                  lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.display_memory()
        self.losses = []

        self.validation_predictions = None
        self.validation_targets = None

    def train(self, dataset: PickingDataset, batch_size: int):

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

    def predict_trace(self, trace: Trace, pick_time: UTCDateTime):

        n_sample = int((pick_time - trace.stats.starttime)
                       * trace.stats.sampling_rate)

        n_sample_start = int(n_sample - self.n_sample / 2)
        n_sample_end = int(n_sample_start + self.n_sample)

        data = trace.data[n_sample_start: n_sample_end]
        data -= np.mean(data)

        self.predict(data)

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

    @classmethod
    def read(cls, file_name):
        with(open(file_name, 'rb')) as f_in:
            return pickle.load(f_in)


def read_picker(file_name):
    return AIPicker.read(file_name)



