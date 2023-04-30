from useis.processors import classifier
from importlib import reload
from pathlib import Path
from uquake.core.inventory import Inventory
reload(classifier)

root_dir = '/data_1/projects/'
project_name = 'classification_3'
network_name = 'OT'

ec = classifier.Classifier2(root_dir, project_name, network_name,
                            sampling_rates=[6000])

if ec.inventory is None:
    inventory = Inventory.from_url(
        'https://www.dropbox.com/s/wmn1u8c93ti94el/inventory.xml?dl=1')
    ec.add_inventory(inventory)

ec.create_training_dataset('/data_1/ot-reprocessed-data/', resume=True,
                           reset_training=False)

# import cProfile
# f = Path('/data_1/ot-reprocessed-data/f7ea5bf15e545fdbb663a2c26f372561.mseed')
# cProfile.run('ec.__process__(f)')

# import cProfile(ec.__process__())