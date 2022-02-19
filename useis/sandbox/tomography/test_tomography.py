from useis.processors import tomography
from importlib import reload

reload(tomography)
tomo = tomography.Tomography('/data_1/projects/tomo_test', 'tomo_test', 'TGT')

# tomo.synthetic()

tomo.__add_random_events__(1000)
tomo.__add_random_travel_times__()

rays = tomo.ray_tracing()
ray = rays[0]


# from uquake.grid.nlloc import VelocityGrid3D
# import numpy as np
# import scipy as sp
# import matplotlib.pyplot as plt
#
# # Grid parameters
# network_code = 'test'
# spacings = [10, 10, 10]
# origin = [0, 0, 0]
# dims = [100, 100, 100]
#
# nsensors = 12
# nevents = 1000
#
# # Building the velocity model
# velocity = VelocityGrid3D(network_code, dims, origin, spacings)
# velocity.fill_random(5000, 200, 5)
#
# # generate random events in the grid
# events = velocity.generate_random_points_in_grid(nevents)
# event_grid = velocity.transform_to_grid(events)
#
# # generate random sensors in the grid
# sensors = velocity.generate_random_points_in_grid(nsensors)
# sensor_grid = velocity.transform_to_grid(sensors)
# sensor_labels = [f'sta{i}' for i in range(0, nsensors)]
#
# times = velocity.to_time_multi_threaded(sensors, sensor_labels)
#
#
# ray = times[0].ray_tracer(events)
# nodes = velocity.transform_to_grid(ray.nodes)
#
# plt.clf()
# plt.imshow(velocity.data[:,:,10].T)
# plt.colorbar()
#
# plt.plot(event_grid[:, 0], event_grid[:, 1], 'k.')
# plt.plot(sensor_grid[:, 0], sensor_grid[:, 1], 'rv')
# plt.plot(nodes[:,0], nodes[:,1], 'r')
#
# plt.show()


