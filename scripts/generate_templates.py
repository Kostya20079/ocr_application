"""Dataset generation script for Template Matching OCR.

Renders standard alphanumeric characters and basic punctuation using TrueType fonts,
trims bounding boxes, and normalizes them onto fixed-size binary template canvases.
Saves template images and an index metadata JSON file.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import string
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.preprocessor import ImagePreprocessor


# Candidate font paths per operating system
LINUX_CANDIDATE_FONTS = [
    # DejaVu (Sans, Serif, Mono)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    # Liberation (Sans, Serif, Mono)
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf",
    # Ubuntu (Sans, Mono)
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf",
    # FreeFont (Sans, Serif, Mono)
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf",
    # Noto
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerif-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf",
    # URW / PostScript standard base fonts (Times, Helvetica, Courier, Palatino, Century equivalents)
    "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Regular.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Bold.otf",
    "/usr/share/fonts/opentype/urw-base35/C059-Roman.otf",
    "/usr/share/fonts/opentype/urw-base35/C059-Bold.otf",
    "/usr/share/fonts/opentype/urw-base35/P052-Roman.otf",
    "/usr/share/fonts/opentype/urw-base35/P052-Bold.otf",
    "/usr/share/fonts/opentype/urw-base35/URWBookman-Demi.otf",
    "/usr/share/fonts/opentype/urw-base35/URWBookman-Light.otf",
    "/usr/share/fonts/opentype/urw-base35/URWGothic-Book.otf",
]

WINDOWS_CANDIDATE_FONTS = [
    # Standard Core Fonts
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "arial.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "arialbd.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "times.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "timesbd.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "calibri.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "calibrib.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "segoeui.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "segoeuib.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "cour.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "courbd.ttf"),
    # Additional Standard Book & Document Fonts
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "georgia.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "georgiab.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "garamond.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "tahoma.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "tahomabd.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "verdana.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "verdanab.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "consola.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "consolab.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "trebuc.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "trebucbd.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "palabi.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "pala.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "palab.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "cambria.ttc"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "comic.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "comicbd.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "corbel.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "corbelb.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "candara.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "candarab.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "constant.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "constantb.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "century.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "bookos.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "bookosb.ttf"),
]

def find_candidate_fonts() -> list[str]:
    """Finds all available candidate fonts across the current operating system."""
    current_os = platform.system().lower()
    candidates = LINUX_CANDIDATE_FONTS if "linux" in current_os else WINDOWS_CANDIDATE_FONTS

    found = [p for p in candidates if os.path.exists(p)]
    if not found:
        default_f = find_default_font()
        return [default_f] if default_f else []
    return found

def find_default_font() -> str:
    """Finds first existing default font across Windows, Linux, and macOS."""
    current_os = platform.system().lower()

    if "windows" in current_os:
        candidates = WINDOWS_CANDIDATE_FONTS + LINUX_CANDIDATE_FONTS
    else:
        candidates = LINUX_CANDIDATE_FONTS + WINDOWS_CANDIDATE_FONTS

    for font_path in candidates:
        if os.path.exists(font_path):
            return font_path

    # Try loading common font names directly through Pillow/Windows font registry
    for name in ("arial.ttf", "calibri.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            ImageFont.truetype(name, 20)
            return name
        except OSError:
            continue

    return ""

# Character sets
BASIC_LATIN_UPPER = string.ascii_uppercase
BASIC_LATIN_LOWER = string.ascii_lowercase
DIGITS = string.digits

# Common punctuation and book typography symbols
PUNCTUATION = ".,!?:;-\"\'()[]{}/\\@#$%&*+=<>_~`^|"
# Additional typography symbols often found in books and texts
EXTENDED_SYMBOLS = "„“”‘’—–…"


def char_to_filename(char: str, font_name: str, index: int = 0) -> str:
    """Creates a filesystem-safe filename for a character template."""
    hex_code = f"{ord(char):04x}"
    clean_font = "".join(c for c in font_name if c.isalnum() or c in ("-", "_"))

    label = char if char.isalnum() else "sym"
    return f"template_{label}_{hex_code}_{clean_font}_{index}.png"


def render_character_glyph(
    char: str,
    font_path: str | None = None,
    font_size: int = 64,
    canvas_size: int = 128,
) -> np.ndarray:
    """Renders a single character glyph on a binary canvas (0=background, 255=foreground text).

    Args:
        char: The character string to render.
        font_path: Path to the TrueType (.ttf) font file. If None or not found, uses default.
        font_size: Font size to render.
        canvas_size: Intermediate canvas dimension for drawing text.

    Returns:
        2D numpy array uint8 representing foreground mask (255 for glyph, 0 for background).
    """
    if font_path and os.path.exists(font_path):
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    # Create grayscale image with white background
    img = Image.new("L", (canvas_size, canvas_size), color=255)
    draw = ImageDraw.Draw(img)

    # Use font bounding box to center character
    bbox = draw.textbbox((0, 0), char, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Calculate offset so glyph is drawn in the middle of canvas
    x = (canvas_size - text_w) // 2 - bbox[0]
    y = (canvas_size - text_h) // 2 - bbox[1]
    draw.text((x, y), char, fill=0, font=font)

    arr = np.array(img, dtype=np.uint8)
    binary_glyph = np.where(arr < 128, 255, 0).astype(np.uint8)

    return binary_glyph


def crop_and_normalize(
    binary_glyph: np.ndarray,
    target_size: tuple[int, int] = (32, 32),
    padding: int = 2,
) -> np.ndarray:
    """Tightly crops foreground pixels and normalizes them onto a target canvas preserving aspect ratio.

    Uses ImagePreprocessor.normalize_character to guarantee 100% consistency between template
    generation and OCR inference.

    Args:
        binary_glyph: 2D uint8 numpy array (255=foreground, 0=background).
        target_size: (width, height) of normalized output canvas.
        padding: Safe margin around the glyph inside target canvas.

    Returns:
        Normalized 2D uint8 numpy array with target_size.
    """
    return ImagePreprocessor.normalize_character(
        binary_glyph,
        target_size=target_size,
        padding=padding,
        is_foreground=True,
    )


def generate_dataset(
    output_dir: Path | str = "data/templates",
    font_paths: list[str] | None = None,
    characters: str | None = None,
    target_size: tuple[int, int] = (32, 32),
) -> list[dict[str, Any]]:
    """Generates normalized template images and saves index metadata.

    Args:
        output_dir: Path to directory where templates will be stored.
        font_paths: List of font paths to use.
        characters: String containing all characters to generate.
        target_size: Target dimension (width, height) for templates.

    Returns:
        List of metadata dictionaries for generated templates.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if characters is None:
        # Letters + digits + punctuation + extended typographical symbols
        characters = (
            BASIC_LATIN_UPPER
            + BASIC_LATIN_LOWER
            + DIGITS
            + PUNCTUATION
            + EXTENDED_SYMBOLS
        )

    if not font_paths:
        font_paths = find_candidate_fonts()

    metadata_list: list[dict[str, Any]] = []

    print(f"Generating templates into: {output_path.resolve()}")
    print(f"Total characters: {len(characters)}")
    print(f"Fonts: {font_paths if font_paths else ['Default Pillow Bitmap']}")

    for font_idx, font_path in enumerate(font_paths or [None]):
        font_name = Path(font_path).stem if font_path else "default"

        for char in characters:
            glyph = render_character_glyph(char, font_path=font_path)
            normalized = crop_and_normalize(glyph, target_size=target_size)

            filename = char_to_filename(char, font_name, font_idx)
            file_dest = output_path / filename

            Image.fromarray(normalized).save(file_dest)

            metadata_list.append(
                {
                    "char": char,
                    "ascii_code": ord(char),
                    "font_name": font_name,
                    "font_path": font_path,
                    "filename": filename,
                    "width": target_size[0],
                    "height": target_size[1],
                }
            )

    # Save index.json
    index_file = output_path / "index.json"
    if index_file.exists():
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                existing_meta = json.load(f)
            generated_filenames = {m["filename"] for m in metadata_list}
            for entry in existing_meta:
                fn = entry.get("filename")
                if fn and fn not in generated_filenames and (output_path / fn).exists():
                    metadata_list.append(entry)
        except Exception:
            pass

    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    print(f"Successfully generated {len(metadata_list)} templates.")
    print(f"Metadata index written to: {index_file}")
    return metadata_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate normalized character templates for Template Matching OCR."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/templates",
        help="Output folder for template images (default: data/templates)",
    )
    parser.add_argument(
        "--fonts",
        "-f",
        nargs="*",
        default=None,
        help="Paths to TTF font files to use.",
    )
    parser.add_argument(
        "--size",
        "-s",
        type=int,
        default=32,
        help="Target square dimension for templates (default: 32 for 32x32)",
    )
    parser.add_argument(
        "--letters-only",
        action="store_true",
        help="Generate only standard basic Latin letters (A-Z, a-z).",
    )
    parser.add_argument(
        "--no-extended-symbols",
        action="store_true",
        help="Exclude extended book typography symbols (e.g. «», —, …, ©, §).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.letters_only:
        char_set = BASIC_LATIN_UPPER + BASIC_LATIN_LOWER
    elif args.no_extended_symbols:
        char_set = BASIC_LATIN_UPPER + BASIC_LATIN_LOWER + DIGITS + PUNCTUATION
    else:
        char_set = (
            BASIC_LATIN_UPPER
            + BASIC_LATIN_LOWER
            + DIGITS
            + PUNCTUATION
            + EXTENDED_SYMBOLS
        )

    generate_dataset(
        output_dir=args.output,
        font_paths=args.fonts,
        characters=char_set,
        target_size=(args.size, args.size),
    )
