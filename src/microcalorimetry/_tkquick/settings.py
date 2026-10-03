"""
Stores information about persistent settings for GUI.
"""

from pathlib import Path
from typing import TypedDict
import yaml

# stores user settings about the program
# this should always just be there
USER_SETTINGS_DIR = Path.home() / '.microcalorimetry'
USER_SETTINGS_DIR.mkdir(exist_ok=True)

# default settings that are stored at the user level
USER_SETTINGS_FILE = USER_SETTINGS_DIR / 'settings.yml'
USER_SETTINGS_DEFAULTS = {'last_cwd': None}

# default settings that are stored at the project level
CWD_SETTINGS_DEFAULTS = {}


def cwd_settings_dir() -> Path:
    """Directory containing consistent settings and informaton."""
    out = Path.cwd() / '.microcalorimetry'
    out.mkdir(exist_ok=True)
    return out


def cwd_history_file() -> Path:
    """File that contains persistent history of function inputs."""
    out = cwd_settings_dir() / 'history.yml'
    return out


def get_user_settings() -> 'Settings':
    """Get the user level settings."""
    return Settings(USER_SETTINGS_FILE, USER_SETTINGS_DEFAULTS)


def get_cwd_settings() -> 'Settings':
    """Get the CWD level settings."""
    return Settings(cwd_settings_dir() / 'settings.yml', CWD_SETTINGS_DEFAULTS)


class Settings(object):
    """Access and stores settings in a .yml file."""

    def __init__(self, file: Path, defaults: dict):
        # start with defaults if setting don't already exist
        self._file = file
        settings = defaults

        # overload with any user settings that we find
        if self._file.exists():
            try:
                with open(self._file, 'r') as f:
                    user_settings = yaml.safe_load(f)
                settings = settings | {
                    k: v for k, v in user_settings.items() if k in defaults
                }
            except Exception as e:
                print('User settings failed to read, defaulting user settings.')
                settings = defaults

        self._settings = settings

        # print(self._settings)
        # dump new settings to the file
        with open(self._file, 'w') as f:
            yaml.safe_dump(self._settings, f)

    # for arbitrary attributes, check inside _settings, grab it from the _settings dictionary
    def __getattr__(self, name: str):
        if name == '_settings':
            raise Exception
        return self._settings[name]

    # attributes are only included in the settings dictionary if they were defined
    # in the defaults
    def __setattr__(self, name: str, value):
        if name in ['_settings', '_file']:
            super(Settings, self).__setattr__(name, value)
        elif name in self._settings:
            self._settings[name] = value
            with open(self._file, 'w') as f:
                yaml.safe_dump(self._settings, f)
        else:
            raise KeyError(f'{name} not in settings.')
