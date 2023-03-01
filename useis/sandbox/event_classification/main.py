from useis.processors.classifier import Classifier
from uquake import read_inventory
from uquake.grid.nlloc import (ModelLayer, LayeredVelocityModel,
                               VelocityGridEnsemble)
import numpy as np

project_code = 'classifier'
network_code = 'OT'


def make_layered_model():

    origin = np.array([650200, 4766170, -500])
    dimensions = np.array([100, 101, 68])
    spacing = np.array([25, 25, 25])

    z = [1168, 459, -300, -500]
    vp_z = [4533, 5337, 5836, 5836]
    vs_z = [2306, 2885, 3524, 3524]

    p_layered_model = LayeredVelocityModel(project_code)
    s_layered_model = LayeredVelocityModel(project_code, phase='S')
    for (z_, vp, vs) in zip(z, vp_z, vs_z):
        layer = ModelLayer(z_, vp)
        p_layered_model.add_layer(layer)
        layer = ModelLayer(z_, vs)
        s_layered_model.add_layer(layer)

    vp_grid_3d = p_layered_model.to_3d_grid(network_code, dimensions, origin,
                                             spacing)
    vs_grid_3d = s_layered_model.to_3d_grid(network_code, dimensions, origin,
                                             spacing)
    return VelocityGridEnsemble(vp_grid_3d, vs_grid_3d)


classifier = Classifier('/data_1/project/', project_code, network_code)

velocities = make_layered_model()

inventory = read_inventory('inventory/OT.xml')
classifier.add_inventory(inventory=inventory)
classifier.add_velocity(velocities['p'], initialize_travel_times=False)
classifier.add_velocity(velocities['s'], initialize_travel_times=True)
