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

classifier_project.train(override_current_model=True)
# classifier_project.train()

# train, test, validation = classifier_project.split_dataset(use_synthetic=False)

torch.save(classifier_project.event_classifier.model.state_dict,
           f'/data_1/ai_models/{datetime.now()}_classification_model.pth')


ec = classifier_project.event_classifier

out_dict = {'model_state': ec.model.state_dict(),
            'out_features': ec.n_features,
            'label_map': ec.label_mapping,
            'learning_rate': ec.learning_rate,
            'model_id': ec.model_id}

pickle.dump(out_dict, open('/data_1/ai_models/test.pickle', 'wb'))

def read_model(input_file):
    in_dict = pickle.load(open('/data_1/ai_models/test.pickle', 'rb'))

    # def __init__(self, n_features: int, label_mapping, gpu: bool = True,
    #              learning_rate: float = 0.001,
    #              model=models.resnet34(), model_id=None):
    ec2 = model.EventClassifier(in_dict['out_features'], in_dict['label_map'], gpu=False,
                                learning_rate=in_dict['learning_rate'],
                                model_id=in_dict['model_id'])
    ec2.model.load_state_dict(in_dict['model_state'])
    ec2.model.eval()



classifier_project
