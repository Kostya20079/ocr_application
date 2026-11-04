"""
Build script to compile the OCR application into a standalone executable (.exe or binary).

Usage:
    python scripts/build_exe.py
"""

import subprocess
import sys
from pathlib import Path

def build() -> None:
    project_root = Path(__file__).resolve().parent.parent
    app_entry = project_root / "app.py"
    data_dir = project_root / "data"
    locales_dir = project_root / "locales"

    separator = ";" if sys.platform.startswith("win") else ":"
    data_arg = f"{data_dir}{separator}data"
    locales_arg = f"{locales_dir}{separator}locales"

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--windowed",
        "--name", "OCRApplication",
        "--add-data", data_arg,
        "--add-data", locales_arg,
        str(app_entry)
    ]

    print("Running PyInstaller:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=project_root)

if __name__ == "__main__":
    build()
