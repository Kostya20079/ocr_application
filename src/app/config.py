from pathlib import Path
from src.utils.helpers import get_resource_path

APP_NAME = "Optical Character Recognition Application"
APP_VERSION = "0.1.0"

# Main resource directories
DATA_DIR: Path = get_resource_path("data")
SAMPLES_DIR: Path = DATA_DIR / "samples"
TEMPLATES_DIR: Path = DATA_DIR / "templates"
MODELS_DIR: Path = DATA_DIR / "models"
LOCALES_DIR: Path = get_resource_path("locales")

DEFAULT_LOCALE = "pl"