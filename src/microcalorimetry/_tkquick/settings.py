"""
Stores information about persistent settings / information about GUI.
"""

from pathlib import Path


def settings_dir():
    """Directory containing consistent settings and informaton."""
    out = Path.cwd() / '.microcalorimetry'
    out.mkdir(exist_ok=True)
    return out


def history_file():
    """File that contains persistent history of function inputs."""
    out = settings_dir() / 'history.yml'
    return out


# default settings will be stored here,
# but these should be overridable by a settings file
# (at some point) if it gets to that point. Probably not.
ADJUSTABLE_SETTINGS = {
    # when opening a new initial director, opens from here
    'DEFAULT_NEW_CWD_INITIAL_DIR': Path.home(),
}
