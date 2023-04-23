from useis.processors import classifier
from importlib import reload
from pathlib import Path
from uquake.core import read, read_events
import numpy as np
reload(classifier)

root_dir = '/data_1/projects/'
project_name = 'classification_2'
network_name = 'test'

ec = classifier.Classifier2('/data_1/projects/', 'classification_2', 'test',
                            sampling_rates=[6000])

records = ec.training_db_manager.filter(categories='noise')

root_dir = Path('/data_1/ot-reprocessed-data/')

filenames = np.unique([record.mseed_file for record in records])

for f in filenames:
    filename = root_dir / f
    st = read(filename)[0:20]
    cat = read_events(filename.with_suffix('.xml'))
    cr = ec.predict(st, event_locaiton=cat[0].origins[-1].loc)
    print(cr)
    # print(cr.predicted_class_ensemble(cat[0].origins[-1].loc))
    st.plot()