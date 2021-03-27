import os
from dynaconf import LazySettings
from pathlib import Path


class Settings(LazySettings):
    def __init__(self, settings_location, settings_file='settings.toml'):
        """
        Init function currently just initializes the object allowing
        """

        config_dir = Path(settings_location)

        dconf = {}
        dconf.setdefault('ENVVAR_PREFIX_FOR_DYNACONF', 'UQUAKE')

        env_prefix = '{0}_ENV'.format(
            dconf['ENVVAR_PREFIX_FOR_DYNACONF']
        )  # SPP_ENV

        dconf.setdefault(
            'ENV_FOR_DYNACONF',
            os.environ.get(env_prefix, 'DEV').upper()
        )

        dconf['SETTINGS_FILE_FOR_DYNACONF'] = os.path.join(settings_location,
                                                           settings_file)
        dconf['ROOT_PATH_FOR_DYNACONF'] = settings_location

        super().__init__(**dconf)

