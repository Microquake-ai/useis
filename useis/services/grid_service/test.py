from useis.services.grid_service import client
from uquake.grid.nlloc import VelocityGrid3D
from uquake.nlloc.nlloc import Srces
from uquake.core import read_inventory
from importlib import reload
reload(client)

root_dir = 'projects'
test_project = 'test_project'
test_network = 'TN'

grid_dim = [10, 10, 10]
grid_spacing = [1, 1, 1]
grid_origin = [0, 0, 0]


def create_velocity():
    p_velocity_grid = VelocityGrid3D(test_network,
                                     grid_dim,
                                     grid_origin,
                                     grid_spacing,
                                     phase='P', value=5000)
    s_velocity_grid = VelocityGrid3D(test_network,
                                     grid_dim,
                                     grid_origin,
                                     grid_spacing,
                                     phase='S', value=3000)

    return p_velocity_grid, s_velocity_grid


p_velocity, s_velocity = create_velocity()
gc = client.GridClient('http://127.0.0.1:8000/', test_project, test_network)

response = gc.add_3d_velocity(p_velocity)
response = gc.add_3d_velocity(s_velocity)
# inv = read_inventory('/home/jpmercier/Repositories/Nanometrics/data'
#                      '/OSA_Light.xml', xy_from_lat_lon=True)
# response = gc.add_inventory(inv)
srces = Srces.generate_random_srces_in_grid(p_velocity,
                                            n_srces=20)
response = gc.add_srces(srces)
response = gc.initialize_travel_times(multi_threaded=True)
# print(gc.list_3d_velocity())

response, points = gc.generate_random_points_in_grid(n_points=1)

x = points[0][0]
y = points[0][1]
z = points[0][2]

response, observations = gc.generate_random_observations(x, y, z)

response = gc.event_location(observations)
