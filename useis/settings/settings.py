import os
from dynaconf import LazySettings
from pathlib import Path

class Settings(LazySettings):
    def __init__(self, settings_location):
        """
        Initializes the settings object, ensuring all .toml files in the specified
        settings_location are read.
        """

        # Define configuration dictionary for Dynaconf
        dconf = {}
        dconf.setdefault('ENVVAR_PREFIX_FOR_DYNACONF', 'USEIS')

        env_prefix = f"{dconf['ENVVAR_PREFIX_FOR_DYNACONF']}_ENV"  # e.g., USEIS_ENV

        # Determine the environment
        dconf.setdefault(
            'ENV_FOR_DYNACONF',
            os.environ.get(env_prefix, 'DEV').upper()
        )

        # List all .toml files in the settings_location
        config_dir = Path(settings_location)
        toml_files = list(config_dir.glob('*.toml'))

        # Ensure the settings.toml is the first file to load if it exists
        settings_file = config_dir / 'settings.toml'
        if settings_file.exists() and settings_file in toml_files:
            toml_files.remove(settings_file)
            toml_files.insert(0, settings_file)

        # Include the list of .toml files in the configuration
        dconf['SETTINGS_FILE_FOR_DYNACONF'] = [str(f) for f in toml_files]
        dconf['ROOT_PATH_FOR_DYNACONF'] = settings_location

        # Initialize the LazySettings with the configuration
        super().__init__(**dconf)


