from ..core.project_manager import *


class NLLOC(ProjectManager):
    def __init__(self, path, project_name, network_code, use_srces=False):

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

        self.last_location = None

        super.__init__(path, project_name, network_code, use_srces=use_srces)

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

    def add_observations(self, observations):
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
                     delete_output_files=True, event=None):

        import subprocess

        if event is not None:
            observations = Observations.from_event(event=event)

        if (observations is None) and (self.observations is None):
            raise ValueError('The current run does not contain travel time'
                             'observations. Observations should be added to '
                             'the current run using the add_observations '
                             'method.')

        elif observations is not None:
            self.add_observations(observations)

        self.write_control_file()

        cmd = ['NLLoc', self.control_file]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)

        logger.info('locating event using NonLinLoc')
        t0 = time()
        output, error = process.communicate()
        t1 = time()
        logger.info(f'done locating event in {t1 - t0:0.2f} seconds')

        t, x, y, z = read_hypocenter_file(self.output_file_path / 'last.hyp')

        scatters = read_scatter_file(self.output_file_path / 'last.scat')

        rays = None
        if calculate_rays:
            logger.info('calculating rays')
            t0_ray = time()
            rays = self.travel_times.ray_tracer(np.array([x, y, z]))
            t1_ray = time()
            logger.info(f'done calculating rays in {t1_ray - t0_ray:0.2f} '
                        f'seconds')

        uncertainty_ellipsoid = calculate_uncertainty(scatters[:, :-1])

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

        result = {'event_id': str(t),
                  'time': t,
                  'hypocenter_location': np.array([x, y, z]),
                  'scatters': scatters,
                  'rays': rays,
                  'uncertainty': uncertainty_ellipsoid}

        self.last_location = result

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