from pathlib import Path
import pickle
from uquake.core.inventory import read_inventory
from uquake.core.logging import logger
from uquake.nlloc import (Srces)
from uquake.core.event import ConfidenceEllipsoid, OriginUncertainty
from uquake.grid import nlloc as nlloc_grid
from uquake.grid import read_grid
from time import time
import numpy as np
import os
import shutil
from ..settings.settings import Settings
from uquake.core.event import AttribDict
from uquake.grid.hdf5 import H5TTable
from .projection import Projection
from pydantic import conlist


def read_srces(fname):
    with open(fname, 'rb') as srces_file:
        return pickle.load(srces_file)


class ProjectManager(object):

    inventory_file_name = 'inventory.xml'

    def __init__(self, base_projects_path, project_name, network_code,
                 use_srces=False, **kwargs):
        """
        Interface to manage project and grids.

        :param base_projects_path: base project path
        :type base_projects_path: str
        :param project_name: project name or id
        :type project_name: str
        :param network_code: network name or id
        :type network_code: str
        :param use_srces: if True use the srces files instead of the the
        inventory file should both files be present (default=False)
        :Example:
        >>> from uquake.grid import nlloc as nlloc_grid
        >>> from uquake.nlloc import nlloc
        # initialize the project with the root path of the project,
        the project and the network names
        >>> project_name = 'test'
        >>> network_code = 'test'
        >>> root_path=''
        >>> pm = nlloc.ProjectManager(root_path, project_name, network_code)
        # this will initialize the project and create a specific directory
        # structure if the directories do not exist.

        # add an inventory file to the project
        >>> path_to_uquake_inventory_file = 'PATH/TO/THE/INVENTORY/FILE.xml'

        ..note :: the uquake inventory object inherit from the obspy inventory
                  and
                  behaves in a very similar way. It, however, differs from
                  its parent has it implements properties specific to local
                  earthquake who are either expressed using UTM coordinates
                  system or a local coordinates system that cannot necessarily
                  be translated to latitude and longitude. Beyond this minor
                  difference, uquake inventory structure also differs from
                  the standard inventory and implements a slightly different
                  hierarchy more representative of mine monitoring systems.
                  The uquake Inventory object hierarchy is as follows:
                 1. Inventory
                    1.1. Networks: A useis network. a network includes at
                                   least one data acquisition station.
                        1.1.1 Stations: A place where the data acquisition is
                                        performed. For instance, station
                                        usually includes power, communication
                                        and data acquisition equipment. One
                                        or more site can be connected to a
                                        station.
                            1.1.1.1 sites: A instrument converting a
                                             physical phenomenon to data either
                                             digital or analog. A site
                                             comprises one or more channel.
                                1.1.1.1.1 Channels: A channel is a the smallest
                                                    unit of measuring.


        >>> inventory = read_inventory(path_to_uquake_inventory_file)
        >>> pm.add_inventory(inventory)

        # alternatively, sites can be added to the using the srces object.
        # sites can be added this way from a nlloc.nlloc.Srces object using
        the .add_srces method.
        ..note:: srces stands for sources and it is the nomenclature used in
                 NonLinLoc. This might be a soure of confusion for the users.
                 In addition, what NonLinLoc refers to as station is called a
                 sites in this context.

        # srces object can be constructed from an inventory as follows:
        >>> srces = Srces.from_inventory(inventory)
        # alternatively, each sites can be added individually using the
        .add_site method. As follows:
        >>> srces = Srces()
        >>> x = 250
        >>> y = 250
        >>> z = 250
        >>> elevation = 0
        >>> srces.add_site('site label', x, y, z, elev=elevation)

        # Srces can be added to the project as follows:
        >>> pm.add_srces(srces)

        # add the velocity models to the project. P- and S-wave velocity models
        can be added separately from a nlloc.grid.VelocityGrid3D
        using the .add_velocity method or from a nlloc.grid.VelocityEnsemble
        object using the .add_velocities method

        >>> origin = [0, 0, 0]
        >>> spacing = [25, 25, 25]
        >>> dimensions = [100, 100, 100]

        >>> nlloc_grid.VelocityGrid3D()
        >>> vp = 5000  # P-wave velocity in m/s
        >>> vs = 3500  # S-wave velocity in m/s
        >>> p_velocity = nlloc_grid.VelocityGrid3D(network_code, dimensions,
        >>>                                        origin, spacing, phase='P',
        >>>                                        value=5000)
        >>> s_velocity = nlloc_grid.VelocityGrid3D(network_code, dimensions,
        >>>                                        origin, spacing, phase='S',
        >>>                                        value=5000 )
        >>> pm.add_velocity(p_velocity)
        >>> pm.add_velocity(s_velocity)
        # Alternatively
        >>> velocities = nlloc_grid.VelocityGridEnsemble(p_velocity,
        >>>                                              s_velocity)
        >>> pm.add_velocities(velocities)
        # Adding a velocity model of the velocity models triggers the
        # calculation of the travel time grids.
        # It is possible to manually trigger the calculation of the travel
        # time grid by invoking
        >>> pm.init_travel_time_grids()
        # this should not, however, be required.

        # prior to running the location, NonLinLoc need to be configured.
        # configuring NonLinLoc can be done using the nlloc.nlloc.NonLinLoc
        # object
        >>> nonlinloc = nlloc.NonLinLoc()
        # this will initialize the nonlinloc object sith default value. Those
        # value have been used to locate useis events in a volumes of
        # approximately 3000 m x 3000 m x 1500 m. The parameters should be
        # provide adequate results for volumes of similar scale but would need
        # to be adapted to smaller or larger volumes.

        """

        self.project_name = project_name
        self.network_code = network_code

        p_vel_base_name = nlloc_grid.VelocityGrid3D.get_base_name(network_code,
                                                                  'P')
        s_vel_base_name = nlloc_grid.VelocityGrid3D.get_base_name(network_code,
                                                                  'S')

        self.base_projects_path = Path(base_projects_path)
        base_project_path = Path(base_projects_path)
        root = base_project_path / project_name / network_code
        self.paths = {'base': base_project_path,
                      'root': base_project_path / project_name / network_code,
                      'archives': base_project_path / 'archives',
                      'inventory': root / 'inventory',
                      'config': root / 'config',
                      'velocities': root / 'velocities',
                      'times': root / 'times',
                      'hdf5_times': root / 'hdf5_times',
                      'ai_models': root / 'ai_models',
                      'debug': root / 'debug',
                      'index': root / 'index',
                      'databases' : root / 'databases'}

        self.paths = AttribDict(self.paths)

        self.files = {'inventory': self.paths.inventory / 'inventory.xml',
                      'srces': self.paths.inventory / 'srces.pickle',
                      'settings': self.paths.config / 'settings.toml',
                      'services_settings': self.paths.config /
                                           'services_settings.toml',
                      'projection_settings': self.paths.config /
                                             'projection_settings.toml',
                      'p_velocity': self.paths.velocities / p_vel_base_name,
                      's_velocity': self.paths.velocities / s_vel_base_name,
                      'hdf5_tt': self.paths.hdf5_times /
                                 'travel_times_tables.h5f',
                      'classification_model': self.paths.ai_models /
                                              'classification.pt',
                      'picker_model': self.paths.ai_models / 'picker.pickle'}

        self.files = AttribDict(self.files)
        self.databases = AttribDict()

        # create the directory if it does not exist
        self.paths.root.mkdir(parents=True, exist_ok=True)
        for key in self.paths.keys():
            path = self.paths[key]
            logger.info(path)
            path.mkdir(parents=True, exist_ok=True)

        # SETTINGS
        settings_path = Path(os.path.realpath(__file__)).parent.parent

        if not self.files.settings.is_file():
            settings_template = settings_path / \
                                    'settings/settings_template.toml'

            shutil.copyfile(settings_template, self.files.settings)

        if not self.files.projection_settings.is_file():
            settings_template = settings_path / \
                                'settings/projection_settings_template.toml'
            shutil.copyfile(settings_template, self.files.projection_settings)

        if not self.files.services_settings.is_file():
            settings_path = Path(os.path.realpath(__file__)).parent.parent
            settings_template = settings_path / \
                                'settings/services_settings_template.toml'

            shutil.copyfile(settings_template, self.files.services_settings)

        self.settings = Settings(str(self.paths.config))

        self.srces = None
        self.inventory = None

        if self.files.inventory.exists():
            self.inventory = read_inventory(str(self.files.inventory))
            self.srces = Srces.from_inventory(self.inventory)
        elif self.files.srces.exists():
            with open(self.files.srces, 'rb') as srces_file:
                self.srces = pickle.load(srces_file)

        self.p_velocity = None
        try:
            fname = str(self.paths.velocities / p_vel_base_name)
            self.p_velocity = read_grid(fname, format='NLLOC')
        except Exception as e:
            logger.warning(e)

        self.s_velocity = None
        try:
            fname = str(self.paths.velocities / s_vel_base_name)
            self.s_velocity = read_grid(fname, format='NLLOC')
        except Exception as e:
            logger.warning(e)

        self.velocities = None
        if (self.p_velocity is not None) and (self.s_velocity is not None):
            self.velocities = nlloc_grid.VelocityGridEnsemble(self.p_velocity,
                                                              self.s_velocity)

        self.paths.times = self.paths.root / 'times'
        self.paths.times.mkdir(parents=True, exist_ok=True)
        file_list = list(self.paths.times.glob('*time*'))

        self.travel_times = None
        try:
            self.travel_times = nlloc_grid.TravelTimeEnsemble.from_files(
                self.paths.times)
        except Exception as e:
            logger.error(e)

    @property
    def htt(self):
        return self.hdf5_tt

    @property
    def hdf5_tt(self):
        return self.travel_times.to_hdf5(self.files.hdf5_tt)

    @property
    def projection(self):
        offset = self.settings.projection.local.offset.to_list()
        return Projection(global_epsg_code=
                          self.settings.projection.GLOBAL.epsg,
                          local_epsg_code=self.settings.projection.local.epsg,
                          offset=offset)

    def set_global_epsg_code(self, epsg_code: int):
        self.settings.projection.GLOBAL.epsg = epsg_code
        self.projection.write(self.files.projection_settings)

    def set_local_epsg_code(self, epsg_code: int):
        self.settings.projection.local.epsg = epsg_code
        self.projection.write(self.files.projection_settings)

    def set_projection_offset(self, offset: conlist(float, min_items=3,
                                                    max_items=3)):
        self.settings.projection.local.offset = offset
        self.projection.write(self.files.projection_settings)

    @property
    def global_epsg_code(self):
        return self.projection.global_epsg_code

    @global_epsg_code.setter
    def global_epsg_code(self, value):
        self.projection.global_epsg_code = value
        self.projection.write(self.files.projection_settings)

    @property
    def local_epsg_code(self):
        return self.projection.local_epsg_code

    @local_epsg_code.setter
    def local_epsg_code(self, value):
        self.projection.local_epsg_code = value
        self.projection.write(self.files.projection_settings)

    @property
    def projection_offset(self):
        return self.projection.offset

    @local_epsg_code.setter
    def local_epsg_code(self, value: conlist(float, min_items=3, max_items=3)):
        self.projection.offset = value

        self.projection.write(self.files.projection_settings)

    def transform_to_global(self, easting: float = 0, northing: float = 0,
                            z: float = 0):
        return self.projection.transform_to_global(easting=easting,
                                                   northing=northing,
                                                   z=z)

    def transform_to_local(self, longitude: float = 0, latitude: float = 0,
                           z: float = 0):
        return self.projection.transform_to_local(longitude=longitude,
                                                  latitude=latitude)

    @staticmethod
    def exists(project_path, project_name, network_code):
        project_path = Path(project_path) / project_name / network_code
        return project_path.exists()

    def init_travel_time_grids(self, multi_threaded=True):
        """
        initialize the travel time grids
        """

        has_velocity = self.has_p_velocity() or self.has_s_velocity()

        abort = not (has_velocity and self.has_inventory())

        if not self.has_p_velocity():
            logger.warning('No P-wave velocity found in project')

        if not self.has_s_velocity():
            logger.warning('No S-wave velocity found in project')

        if not self.has_inventory():
            logger.warning('No inventory file found in project')

        if abort:
            logger.warning('travel time grids will not be calculated')
            return

        logger.info('initializing the travel time grids')
        t0 = time()
        seeds = self.srces.locs
        seed_labels = self.srces.labels

        tt = self.velocities.to_time(seeds, seed_labels,
                                     multi_threaded=multi_threaded)

        self.travel_times = tt

        for fle in self.paths.times.glob('*time*'):
            fle.unlink(missing_ok=True)

        self.travel_times.write(self.paths.times)
        self.travel_times.write_hdf5(self.files.hdf5_tt)
        t1 = time()
        logger.info(f'done initializing the travel time grids in '
                    f'{t1 - t0:0.2f} seconds')

    def add_inventory(self, inventory, create_srces_file: bool = True,
                      initialize_travel_time: bool = True):
        """
        adding a inventory object to the project
        :param inventory: station xml inventory object
        :type inventory: uquake.core.inventory.Inventory
        :param create_srces_file: if True create or replace the srces file
        (Default: True)
        :param initialize_travel_time: if True the travel time grids are
        initialized (default: True)
        :return:
        """

        inventory.write(str(self.files.inventory))
        self.srces = Srces.from_inventory(inventory)
        self.inventory = inventory
        if create_srces_file:
            with open(self.files.srces, 'wb') as srces_file:
                pickle.dump(self.srces, srces_file)

        if initialize_travel_time:
            self.init_travel_time_grids()
        else:
            logger.warning('the travel time grids will not be initialized, '
                           'the inventory and the travel time grids might '
                           'be out of sync. To initialize the travel time '
                           'grids make sure initialize_travel_time is set to '
                           'True')

    def add_inventory_from_file(self, inventory_file: str, format='StationXML',
                                create_srces_file: bool = True,
                                initialize_travel_time: bool = True,
                                xy_from_lat_lon: bool = False,
                                input_projection=4326,
                                output_projection=None):
        """
        adding a inventory object to the project from a file
        :param inventory_file: station xml inventory object
        :type inventory_file: uquake.core.inventory.Inventory
        :param format: file format for the inventory file
        :param create_srces_file: if True create or replace the srces file
        (Default: True)
        :param initialize_travel_time: if True the travel time grids are
        initialized (default: True)
        :param xy_from_lat_lon: convert latitude and longitude in the file to
        cartesian coordinates
        :param input_projection: input projection (default=4326 for typical
        latitude and longitude)
        :param output_projection: output projection (default None)
        :return:
        """

        if output_projection is not None:
            inventory = read_inventory(inventory_file, format=format,
                                       xy_from_lat_lon=xy_from_lat_lon,
                                       input_projection=input_projection,
                                       output_projection=output_projection)
        else:
            inventory = read_inventory(inventory_file, format=format,
                                       xy_from_lat_lon=xy_from_lat_lon,
                                       input_projection=input_projection)

        self.add_inventory(inventory, create_srces_file=create_srces_file,
                           initialize_travel_time=initialize_travel_time)

    def add_srces(self, srces, force=False, initialize_travel_time: bool=True,
                  multi_threaded: bool=True):
        """
        add a list of sources to the projects
        :param srces: list of sources or sites
        :param force: force the insertion of the srces object if an inventory
        file is present
        :param initialize_travel_time: if True, initialize the travel time
        grid
        :type srces: Srces
        :param multi_threaded: if true, use multi-threading
        :type multi_threaded: bool

        ..warning:: travel time should be initialized when the sites/srces
        are updated. Not doing so, may cause the sites/source and the
        travel time grids to be incompatible.
        """

        if not isinstance(srces, Srces):
            raise TypeError(f'Expecting type {Srces}, given '
                            f'{type(srces)}')

        if self.inventory is not None:
            logger.warning('The project already has an inventory file!')
            if not force:
                logger.warning('exiting...')
                return
            else:
                logger.warning('the force flag value is True, srces object '
                               'will be added.')

        self.srces = srces
        with open(self.files.srces, 'wb') as srces_file:
            pickle.dump(self.srces, srces_file)

        if initialize_travel_time:
            self.init_travel_time_grids(multi_threaded=multi_threaded)
        else:
            logger.warning('the travel time grids will not be initialized, '
                           'the inventory and the travel time grids might '
                           'be out of sync. To initialize the travel time '
                           'grids make sure initialize_travel_time is set to '
                           'True')

    def add_velocities(self, velocities, initialize_travel_times=False):
        """
        add P- and S-wave velocity models to the project
        :param velocities: velocity models
        :type velocities: nlloc.grid.VelocityEnsemble
        :param initialize_travel_times: if True, initialize the travel time
        grid

        ..warning:: travel time should be initialized when the sites/srces
        are updated. Not doing so, may cause the sites/source and the
        travel time grids to be incompatible.
        """

        # velocities.write(path=self.paths.velocities)

        self.velocities = velocities

        for key in velocities.keys():
            self.add_velocity(velocities[key])

        if initialize_travel_times:
            self.init_travel_time_grids()

    def velocity_versioning_handler(self, phase):
        """
        returns the version number of the to be archived velocity model
        :return: int
        """
        file_list = [fle for fle in self.paths.archives.glob(f'*.{phase}.*')]
        # file_list = os.listdir(self.paths.archives / )
        if not file_list:
            return 1

        exts = []
        for element in file_list:
            ext = int(element.split('_')[-1])
            exts.append(ext)

        return np.max(exts) + 1

    def add_velocity(self, velocity, initialize_travel_times=False):
        """
        add P- or S-wave velocity model to the project
        :param velocity: p-wave velocity model
        :type velocity: uquake.grid.nlloc.VelocityGrid3D
        :param initialize_travel_times: if true initialize the travel time grids

        ..warning:: travel time should be initialized when the sites/srces
        are updated. Not doing so, may cause the sites/source and the
        travel time grids to be incompatible.
        """

        if velocity.phase.upper() == 'P':
            self.p_velocity = velocity
        elif velocity.phase.upper() == 'S':
            self.s_velocity = velocity

        velocity.write(self.paths.velocities)

        if initialize_travel_times:
            try:
                self.init_travel_time_grids()
            except Exception as e:
                logger.warning(e)

    def list_active_projects(self):
        """
        List all active projects
        :return: list of active projects
        """
        projects = []
        for project in self.base_projects_path.glob('*'):
            if project.name == 'archives':
                continue
            if project.is_dir():
                projects.append(project.name)

        return projects

    def list_active_networks_project(self, project=None):
        """
        List all active networks
        :param project: the project for which to list the networks
        :return: a list of networks
        """
        if project is None:
            project = self.project_name

        project_directory = self.base_projects_path / project

        if not project_directory.exists():
            logger.warning(f'the project {project} does not exists')
            return

        networks = []
        for network in project_directory.glob('*'):
            if not network.is_dir():
                continue
            networks.append(network.name)

        return networks

    def list_all_active_networks(self):
        """
        list all active projects and networks
        :return: a dictionary of list
        """

        out_dict = {}
        for project in self.list_active_projects():
            networks = self.list_active_networks_project(project=project)
            out_dict[project] = networks

        return out_dict

    def list_all_archived_networks(self):
        """
        List all archived projects and networks
        :return: dictionary of list where the keys are the projects name
        """

        out_dict = {}
        for project in self.paths.archives.glob('*'):
            if not project.is_dir():
                continue
            archived_project_dir = self.paths.archives / project.name
            out_dict[project.name] = []
            for network in archived_project_dir.glob('*'):
                if not network.is_dir():
                    continue
                out_dict[project.name].append(network.name)

        return out_dict

    def has_p_velocity(self):
        if self.p_velocity:
            return True
        return False

    def has_s_velocity(self):
        if self.s_velocity:
            return True
        return False

    def has_inventory(self):
        if self.inventory or self.srces:
            return True
        return False

    def clean_stream(self, stream):
        if 'black_list'.upper() in self.settings.keys():
            for sensor in self.settings.black_list:
                station = sensor.split('.')[0]
                location = sensor.split('.')[1]
                for tr in stream.select(station=station, location=location):
                    stream.remove(tr)

        return stream


