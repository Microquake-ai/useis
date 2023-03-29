from useis.processors import classifier
from importlib import reload
from useis.ai.model import EventClassifier
reload(classifier)

classifier_project = classifier.Classifier('/data_1/projects/', 'classifier', 'OT',
                                           reset_training=False)

train, test, validation = classifier_project.split_dataset(use_synthetic=False)

ec = EventClassifier(train.num_classes)

