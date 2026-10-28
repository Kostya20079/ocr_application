import sys
from pathlib import Path

def get_resource_path(relative_path: str | Path) -> Path:
    """Returns the absolute path to a resource, working both in dev mode and PyInstaller bundles."""
    if hasattr(sys, '_MEIPASS'):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).resolve().parent.parent.parent
    return base_path / relative_path
