from useismic.processors import nlloc
from importlib import reload
from pathlib import Path
from uquake.grid.nlloc import (ModelLayer, LayeredVelocityModel,
                               VelocityGridEnsemble)
from uquake.core import read_events, read_inventory
from uquake.nlloc import Observations
import numpy as np

reload(nlloc)

project_path = Path('/data_2/projects')
project_name = 'test_nlloc'
network = 'nlloc_test'

nlloc = nlloc.NLLOC(project_path, project_name, network)
settings = nlloc.settings

test_artifact_path = Path(nlloc.settings.TEST_ARTIFACTS)
inventory = read_inventory(str(test_artifact_path / 'inventory.xml'))


def get_catalog():
    event_file = test_artifact_path / 'event_file.xml'
    cat = read_events(str(event_file))
    for i, pick in enumerate(cat[0].picks):
        for sensor in inventory.sensors:
            if sensor.alternate_code == pick.waveform_id.station_code:
                cat[0].picks[i].waveform_id.network_code = inventory[0].code
                cat[0].picks[i].waveform_id.station_code = sensor.station.code
                cat[0].picks[i].waveform_id.location_code = \
                    sensor.location_code
                cat[0].picks[i].waveform_id.channel_code = \
                    sensor.channels[0].code
                break
    return cat


event = get_catalog()


def make_layered_model():

    # The origin is the lower left corner
    project_code = project_name
    network_code = network
    origin = np.array(settings.default.grids.origin)
    dimensions = np.array(settings.default.grids.dimensions)
    spacing = np.array(settings.default.grids.spacing)

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

    vp_grid_3d = p_layered_model.gen_3d_grid(network_code, dimensions, origin,
                                             spacing)
    vs_grid_3d = s_layered_model.gen_3d_grid(network_code, dimensions, origin,
                                             spacing)
    velocities = VelocityGridEnsemble(vp_grid_3d, vs_grid_3d)

    return velocities


velocities = make_layered_model()

nlloc.add_inventory(inventory, initialize_travel_time=False)
nlloc.add_velocities(velocities, initialize_travel_time=True)

observations = Observations.from_event(event)

loc = nlloc.run_location(observations=observations, calculate_rays=True)