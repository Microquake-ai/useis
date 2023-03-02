from useis.processors.classifier import Classifier
from uquake import read_inventory
from .velocity import make_layered_model

project_code = 'classifier'
network_code = 'OT'


classifier = Classifier('/data_1/project/', project_code, network_code)


inventory = read_inventory('inventory/OT.xml')
classifier.add_inventory(inventory=inventory)

# optional
# velocities = make_layered_model()
# classifier.add_velocity(velocities['p'], initialize_travel_times=False)
# classifier.add_velocity(velocities['s'], initialize_travel_times=True)
