"""
Special datatypes that can be used with the GUI form fields.
"""

from pathlib import Path


class Folder(Path):
    """Class that identifies function inputs that should be paths to folders for GUI."""


if __name__ == '__main__':
    print(Folder('myfolder'))
