from useis.processors import classifier
from importlib import reload
reload(classifier)

base_directory = '.'
project = 'test_project'
network = 'test_network'

event_classifier = classifier.EventClassifier(base_directory, project, network)

# event_classifier.add_model()