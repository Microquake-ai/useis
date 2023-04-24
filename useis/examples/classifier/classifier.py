from useis.processors import classifier
from importlib import reload
reload(classifier)

base_directory = '.'
project = 'test_project'
network = 'test_network'

event_classifier = classifier.Classifier(base_directory, project, network)

reload(model)

if event_classifier.event_classifier is None:
    classifier_model = model.EventClassifier.load()
    event_classifier.add_model(classifier_model, gpu=False)

# event_classifier.add_model()