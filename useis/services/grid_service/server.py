from fastapi import FastAPI, Depends, File, UploadFile, Query
from typing import Optional, List
from io import BytesIO
import pickle
from loguru import logger
from useis.processors.nlloc import NLLOC
from uquake.core import read_inventory
from uquake.nlloc.nlloc import Observations
from uquake.grid.nlloc import VelocityGrid3D
import os
import json
from fastapi.security import OAuth2PasswordBearer
import numpy as np
from useis.services.models import nlloc
from useis.services.models import grid
from pathlib import Path

app = FastAPI()
root_dir = Path(os.environ.setdefault('nll_base_path', 'projects/'))

# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def deserialize_object(file_obj):
    f_in = BytesIO(file_obj)
    return pickle.load(f_in)


@app.post("/velocity/{project}/{network}/add/3D/", status_code=201)
async def add_velocity_3d(project: str, network: str,
                          velocity: grid.VelocityGrid3D,
                          initialize_travel_times: Optional[bool] = False):

    nll = NLLOC(root_dir, project, network)

    logger.info(f'adding 3D velocity model to the project:{project} and '
                f'network: {network}')
    velocity_grid = velocity.to_uquake(network)
    return nll.add_velocity(velocity_grid,
                            initialize_travel_times=initialize_travel_times)


@app.post("/inventory/{project}/{network}/simple", status_code=201)
async def add_simple_inventory(project: str, network: str, sites: nlloc.Srces):
    """
       Add or update the inventory using a simple site description

       - **sites**: a list of sites with coordinates and labels
       """
    return


@app.post("/inventory/{project}/{network}", status_code=201)
async def add_inventory(project: str, network: str,
                        inventory: bytes = File(...),
                        initialize_travel_times: Optional[bool] = False):
    nll = NLLOC(root_dir, project, network)
    f_in = BytesIO(inventory)
    inv = read_inventory(f_in)
    logger.info(f'adding inventory to {project}/{network}')
    nll.add_inventory(inv, initialize_travel_time=initialize_travel_times)
    return


@app.post("/srces/{project}/{network}", status_code=201)
async def add_srces(project: str, network: str, srces: bytes = File(...),
                    initialize_travel_times: Optional[bool] = False):
    nll = NLLOC(root_dir, project, network)
    sensors = deserialize_object(srces)
    logger.info(f'adding srces to {project}/{network}')
    nll.add_srces(sensors, force=True,
                  initialize_travel_time=initialize_travel_times)
    return


@app.post('/event/locate/{project}/{network}/',
          response_model=nlloc.NLLOCResults)
async def event_location(project: str, network: str,
                         observations: nlloc.Observations):
    nll = NLLOC(root_dir, project, network)
    event = nll.run_location(observations.to_uquake(), calculate_rays=False)
    print(event.loc)
    return 'nanana'


@app.get('/travel_times/{project}/{network}/init')
async def init_travel_times(project: str, network: str,
                            multi_threaded: Optional[bool] = False):
    nll = NLLOC(root_dir, project, network)
    if multi_threaded:
        thread = 'multi thread'
    else:
        thread = 'single thread'
    logger.info(f'calculating the travel time using {thread}')
    nll.init_travel_time_grids(multi_threaded=multi_threaded)
    return


@app.get("/velocity/{project}/{network}")
async def list_velocity(project, network):
    return project, network


@app.get("/test/{project}/{network}/random_locations")
async def generate_random_locations(project: str, network: str,
                                    n_points: Optional[int] = 1):
    """
    Generate random location(s) in grid

    - **project**: project name
    - **network**: network id
    - **n_points**:  number of points random location to return (Default 1)

    the number of points is controlled by the n_points parameter which
    is set to 1 by default.
    """
    nll = NLLOC(root_dir, project, network)
    logger.info(nll.p_velocity.generate_random_points_in_grid())
    logger.info(n_points)
    observation = nll.p_velocity.generate_random_points_in_grid(
        n_points=n_points)
    return json.dumps(list(observation.ravel()))


@app.get("/test/{project}/{network}/observations",
         response_model=nlloc.Observations)
async def generate_random_observations(project: str, network: str,
                                       x: float, y: float, z: float) \
        -> nlloc.Observations:
    """
    Generate random observation in grid

    - **project**: project name
    - **network**: network id
    - **x**: X coordinates at which to calculate the observations
    - **y**: Y coordinates at which to calculate the observations
    - **z**: Z coordinates at which to calculate the observations
    """
    nll = NLLOC(root_dir, project, network)
    e_loc = np.array([x, y, z])
    print(e_loc)
    observations = Observations.generate_observations_event_location(
        nll.travel_times, e_loc=e_loc)

    print(nlloc.Observations.from_uquake(observations))

    return nlloc.Observations.from_uquake(observations)


@app.get('/projects')
async def get_project_list() -> List[str]:
    projects = [str(f).split(os.path.sep)[-1]
                for f in root_dir.glob('*') if f.is_dir()]
    return projects


@app.get('/network/{project}')
async def get_network_list(project: str) -> List[str]:
    networks = [str(f).split(os.path.sep)[-1]
                for f in root_dir.glob(f'{project}/*') if f.is_dir()]
    return networks




