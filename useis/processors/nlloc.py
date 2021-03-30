from ..core.project_manager import *
from uquake.nlloc.nlloc import *
from uquake.core.event import (Catalog, Event, RayCollection, Pick,
                               CreationInfo, Ray)
from uquake.core import UTCDateTime
import numpy as np


def calculate_uncertainty(point_cloud):
    """
    :param point_cloud: location point cloud
    :return: obspy.core.event.OriginUncertainty()
    """

    v, u = np.linalg.eig(np.cov(point_cloud.T))

    major_axis_index = np.argmax(v)

    uncertainty = np.sort(np.sqrt(v))[-1::-1]

    h = np.linalg.norm(u[major_axis_index, :-1])
    vert = u[major_axis_index, -1]

    major_axis_plunge = np.arctan2(-vert, h)
    x = u[major_axis_index, 0]
    y = u[major_axis_index, 1]
    major_axis_azimuth = np.arctan2(x, y)
    major_axis_rotation = 0

    ce = ConfidenceEllipsoid(semi_major_axis_length=uncertainty[0],
                             semi_intermediate_axis_length=uncertainty[1],
                             semi_minor_axis_length=uncertainty[2],
                             major_axis_plunge=major_axis_plunge,
                             major_axis_azimuth=major_axis_azimuth,
                             major_axis_rotation=major_axis_rotation)

    return OriginUncertainty(confidence_ellipsoid=ce,
                             preferred_description='confidence ellipsoid',
                             confidence_level=68)


class NLLOCResult(object):
    def __init__(self, hypocenter: np.array, event_time: UTCDateTime,
                 scatter_cloud: np.ndarray, rays: list,
                 observations: Observations, evaluation_mode: str,
                 evaluation_status: str):
        self.hypocenter = hypocenter
        self.event_time = event_time
        self.scatter_cloud = scatter_cloud
        self.rays = rays
        self.observations = observations
        self.evaluation_mode = evaluation_mode
        self.evaluation_status = evaluation_status

        self.uncertainty_ellipsoid = calculate_uncertainty(
            self.scatter_cloud[:, :-1])

        self.creation_info = CreationInfo(author='uQuake-nlloc',
                                          creation_time=UTCDateTime.now())

    def __repr__(self):
        ce = self.uncertainty_ellipsoid.confidence_ellipsoid
        out_str = f"""
        time (UTC)  : {self.t}
        location    : x- {self.x:>10.1f} (m)
                      y- {self.y:>10.1f} (m)
                      z- {self.z:>10.1f} (m)
        uncertainty : {ce.semi_major_axis_length:0.1f} (1 std - m)
        """
        return out_str

    def __add__(self, other: Event):
        if not (isinstance(other, Catalog) | isinstance(other, Event)):
            raise TypeError(f'object type {type(Event)} or'
                            f'{type(Catalog)} expected. Object of type '
                            f'{type(other)} provided')
        return self.append_to_event(other)

    @property
    def loc(self):
        return self.hypocenter

    @property
    def x(self):
        return self.loc[0]

    @property
    def y(self):
        return self.loc[1]

    @property
    def z(self):
        return self.loc[2]

    @property
    def t(self):
        return self.event_time

    @property
    def arrivals(self) -> list:
        arrivals = []
        for pick in self.observations.picks:
            travel_time = pick.time - self.t
            phase = pick.phase_hint
            sensor_code = pick.sensor
            for ray in self.rays:
                if (ray.sensor_code == sensor_code) & (ray.phase == phase):
                    break
            distance = ray.length

            time_residual = Arrival.calculate_time_residual(ray.travel_time,
                                                            travel_time)

            time_weight = 1
            azimuth = ray.azimuth
            takeoff_angle = ray.takeoff_angle

            arrival = Arrival(phase=phase, distance=distance,
                              time_residual=time_residual,
                              time_weight=time_weight, azimuth=azimuth,
                              takeoff_angle=takeoff_angle,
                              pick_id=pick.resource_id)

            arrivals.append(arrival)

        return arrivals

    @property
    def origin(self):
        origin = Origin(x=self.x, y=self.y, z=self.z, time=self.t,
                        evaluation_mode=self.evaluation_mode,
                        evaluation_status=self.evaluation_status,
                        epicenter_fixed=False, method_id='uQuake-NLLOC',
                        creation_info=self.creation_info,
                        arrivals=self.arrivals,
                        origin_uncertainty=self.uncertainty_ellipsoid,
                        rays=self.rays)
        return origin

    @property
    def event(self):
        return self.export_as_event()

    def append_to_event(self, event: Event) -> Event:
        o = self.origin
        if isinstance(event, Catalog):
            event[0].append_origin_as_preferred_origin(o)
        elif isinstance(event, Event):
            event.append_origin_as_preferred_origin(o)
        return event

    def export_as_event(self):
        o = self.origin
        e = Event(origins=[o], picks=self.observations.picks)
        e.preferred_origin_id = o.resource_id
        return e


class NLLOC(ProjectManager):
    def __init__(self, path: Path, project_name: str, network_code: str,
                 use_srces: bool=False):

        super().__init__(path, project_name, network_code, use_srces=use_srces)

        self.run_id = str(uuid4())
        self.current_run_directory = self.root_directory / 'run' / self.run_id
        self.current_run_directory.mkdir(parents=True, exist_ok=False)

        self.output_file_path = self.current_run_directory / 'outputs'
        self.output_file_path.mkdir(parents=True, exist_ok=True)

        self.observation_path = self.current_run_directory / 'observations'
        self.observation_path.mkdir(parents=True, exist_ok=True)
        self.observation_file_name = 'observations.obs'
        self.observation_file = self.observation_path / \
                                self.observation_file_name
        self.observations = None

        self.template_directory = self.root_directory / 'templates'

        self.template_directory.mkdir(parents=True, exist_ok=True)

        self.template_ctrl_file = self.template_directory / \
                                  'ctrl_template.pickle'

        if self.template_ctrl_file.exists():
            with open(self.template_ctrl_file, 'rb') as template_ctrl:
                self.control_template = pickle.load(template_ctrl)
        else:
            self.control_template = None
            self.add_template_control()

        self.control_file = self.current_run_directory / 'run.nll'

        self.last_event_hypocenter = None
        self.last_event_time = None

        self.nlloc_settings_file = self.config_location / 'nlloc.toml'

        if not self.settings_file.is_file():
            settings_template = Path(os.path.realpath(__file__)).parent / \
                                    '../settings/nlloc_settings_template.toml'

            shutil.copyfile(settings_template, self.nlloc_settings_file)

        self.settings = Settings(str(self.config_location))

    def add_template_control(self, control=Control(message_flag=1),
                             transformation=GeographicTransformation(),
                             locsig=None, loccom=None,
                             locsearch=LocSearchOctTree.init_default(),
                             locmeth=LocationMethod.init_default(),
                             locgau=GaussianModelErrors.init_default(),
                             locqual2err=LocQual2Err(0.0001, 0.0001, 0.0001,
                                                     0.0001, 0.0001),
                             **kwargs):

        if not isinstance(control, Control):
            raise TypeError(f'control is type {type(control)}. '
                            f'control must be type {Control}.')

        if not issubclass(type(transformation), GeographicTransformation):
            raise TypeError(f'transformation is type {type(transformation)}. '
                            f'expecting type'
                           f'{GeographicTransformation}.')

        if not locsearch.type == 'LOCSEARCH':
            raise TypeError(f'locsearch is type {type(locsearch)}'
                            f'expecting type '
                            f'{LocSearchGrid} '
                            f'{LocSearchMetropolis}, or '
                            f'{LocSearchOctTree}.')

        if not isinstance(locmeth, LocationMethod):
            raise TypeError(f'locmeth is type {type(locmeth)}, '
                            f'expecting type {LocationMethod}')

        if not isinstance(locgau, GaussianModelErrors):
            raise TypeError(f'locgau is type {type(locgau)}, '
                            f'expecting type {GaussianModelErrors}')

        if not isinstance(locqual2err, LocQual2Err):
            raise TypeError(f'locqual2err is type {type(locqual2err)}, '
                            f'expecting type {LocQual2Err}')

        dict_out = {'control': control,
                    'transformation': transformation,
                    'locsig': locsig,
                    'loccom': loccom,
                    'locsearch': locsearch,
                    'locmeth': locmeth,
                    'locgau': locgau,
                    'locqual2err': locqual2err}

        with open(self.template_ctrl_file, 'wb') as template_ctrl:
            pickle.dump(dict_out, template_ctrl)

        self.control_template = dict_out

    def write_control_file(self):
        with open(self.control_file, 'w') as control_file:
            control_file.write(self.control)

    def __add_observations__(self, observations):
        """
        adding observations to the project
        :param observations: Observations
        """
        if not isinstance(observations, Observations):
            raise TypeError(f'observations is type {type(observations)}. '
                            f'observations must be type {Observations}.')
        self.observation_path.mkdir(parents=True, exist_ok=True)
        observations.write(self.observation_file_name,
                           path=self.observation_path)

        self.observations = observations

    def run_location(self, observations=None, calculate_rays=True,
                     delete_output_files=True, event=None,
                     evaluation_mode: str = 'automatic',
                     evaluation_status: str = 'preliminary',
                     multithreading=False):

        import subprocess

        if event is not None:
            observations = Observations.from_event(event=event)

        if (observations is None) and (self.observations is None):
            raise ValueError('The current run does not contain travel time'
                             'observations. Observations should be added to '
                             'the current run using the add_observations '
                             'method.')

        elif observations is not None:
            self.__add_observations__(observations)

        self.write_control_file()

        cmd = ['NLLoc', self.control_file]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)

        logger.info('locating event using NonLinLoc')
        t0 = time()
        output, error = process.communicate()
        t1 = time()
        logger.info(f'done locating event in {t1 - t0:0.2f} seconds')

        if event is not None:
            event_time = event.time
        else:
            p_times = []
            for pick in observations.picks:
                p_times.append(pick.time)
            event_time = np.min(p_times)

        if not (self.output_file_path / 'last.hyp').exists():
            logger.error(f'event location failed for event {event_time}!')
        t, x, y, z = read_hypocenter_file(self.output_file_path / 'last.hyp')

        scatters = read_scatter_file(self.output_file_path / 'last.scat')

        rays = None
        if calculate_rays:
            logger.info('calculating rays')
            t0_ray = time()
            rays = self.travel_times.ray_tracer(np.array([x, y, z]),
                                                multithreading=multithreading)
            t1_ray = time()
            logger.info(f'done calculating rays in {t1_ray - t0_ray:0.2f} '
                        f'seconds')

        if delete_output_files:
            for fle in self.observation_path.glob('*'):
                fle.unlink()
            self.observation_path.rmdir()

            output_dir = self.output_file_path

            for fle in output_dir.glob('*'):
                fle.unlink()
            output_dir.rmdir()

            for fle in self.output_file_path.parent.glob('*'):
                fle.unlink()

            self.output_file_path.parent.rmdir()

        result = NLLOCResult(np.array([x, y, z]), t, scatters, rays,
                             observations, evaluation_mode, evaluation_status)

        return result

    def rays(self, hypocenter_location):
        return self.travel_times.ray_tracer(hypocenter_location)

    @property
    def nlloc_files(self):
        return NllocInputFiles(self.observation_file,
                               self.travel_time_grid_location /
                               self.network_code,
                               self.output_file_path / self.network_code)

    @property
    def control(self):

        if self.srces is None:
            raise ValueError('The project does not contain sensors or '
                             'inventory. Sensors (srces) or inventory '
                             'information can be added using the add_srces or'
                             'add_inventory methods.')

        if self.observations is None:
            raise ValueError('The current run does not contain travel time'
                             'observations. Observations should be added to '
                             'the current run using the add_observations '
                             'method.')

        observations = str(self.observations)

        ctrl = ''
        ctrl += str(self.control_template['control']) + '\n'
        ctrl += str(self.control_template['transformation']) + '\n\n'

        if self.control_template['locsig'] is not None:
            ctrl += self.control_template['locsig']

        if self.control_template['loccom'] is not None:
            ctrl += self.control_template['loccom'] + '\n'

        ctrl += str(self.srces) + '\n'

        ctrl += str(self.nlloc_files) + '\n'

        ctrl += str(self.control_template['locsearch'])
        ctrl += str(self.control_template['locmeth'])
        ctrl += str(self.control_template['locgau']) + '\n'

        ctrl += str(self.control_template['locqual2err'])

        if self.p_velocity is not None:
            ctrl += str(LocGrid.init_from_grid(self.p_velocity))
        else:
            raise ValueError('Cannot initialize the LocGrid, the velocity '
                             'grids are not defined')

        return ctrl