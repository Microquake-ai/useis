from useis.processors import classifier
from importlib import reload
from useis.ai import model
from uquake.core.logging import logger
from tqdm import tqdm
import numpy as np
import torch
from torchvision import models
import matplotlib.pyplot as plt
from uquake import read
reload(classifier)
reload(model)

torch.cuda.empty_cache()

classifier_project = classifier.Classifier('/data_1/projects/', 'classifier', 'OT',
                                           reset_training=False)

# train, test, validation = classifier_project.split_dataset(use_synthetic=False)

