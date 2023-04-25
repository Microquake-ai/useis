from useis.processors import classifier
from importlib import reload
from useis.ai import model
from uquake.core import read
from uquake.core.util.requests import download_file_from_url
from time import time
reload(classifier)

base_directory = '.'
project = 'test_project'
network = 'test_network'

event_classifier = classifier.Classifier2(base_directory, project, network)

reload(model)

if event_classifier.event_classifier is None:
    classifier_model = model.EventClassifier2.load()
    event_classifier.add_model(classifier_model)

test_seismic_event_url = \
    'https://www.dropbox.com/s/ld78tj7gtoebxuj/seismic_event_1.mseed?dl=1'
# test_blast_url =
seismic_event_mseed_io = download_file_from_url(test_seismic_event_url)
# blast_mseed_io = download_file_from_url()

st_event = read(seismic_event_mseed_io)
# st_blast = read(blast_mseed_io)

t0 = time()
print(event_classifier.predict(st_event))
t1 = time()

print(f'{t1 - t0}')

# event_classifier.add_model()