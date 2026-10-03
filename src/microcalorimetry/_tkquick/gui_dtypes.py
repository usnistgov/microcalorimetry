"""
Special datatypes that can be used with the GUI form fields.
"""

from pathlib import Path
from typing import Literal, TypeAlias


__all__ = ['Folder', 'SaveAsPath']


class Folder(Path):
    """Identifies function inputs that should be paths to folders."""


class SaveAsPath:
    """Indicate paths to files being read from."""


if __name__ == '__main__':
    print(Folder('myfolder'))
