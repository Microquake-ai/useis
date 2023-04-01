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

# classifier_project.train()
st = read('/data_1/ot-reprocessed-data/177f51f6d4485d2ab4396ec7b02799f8.context_mseed')

output = classifier_project.predict(st)