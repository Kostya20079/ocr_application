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
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Candidate font paths per operating system
LINUX_CANDIDATE_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
]

WINDOWS_CANDIDATE_FONTS = [
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "arial.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "times.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "calibri.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "segoeui.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "cour.ttf"),
]

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
EXTENDED_SYMBOLS = "«»„“”‘’—–…§©®°±×÷"


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

    Args:
        binary_glyph: 2D uint8 numpy array (255=foreground, 0=background).
        target_size: (width, height) of normalized output canvas.
        padding: Safe margin around the glyph inside target canvas.

    Returns:
        Normalized 2D uint8 numpy array with target_size.
    """
    target_w, target_h = target_size
    fg_indices = np.argwhere(binary_glyph > 0)

    # If completely empty glyph (e.g. whitespace)
    if fg_indices.size == 0:
        return np.zeros((target_h, target_w), dtype=np.uint8)

    y_min, x_min = fg_indices.min(axis=0)
    y_max, x_max = fg_indices.max(axis=0) + 1  # slice upper bound is exclusive

    cropped = binary_glyph[y_min:y_max, x_min:x_max]
    crop_h, crop_w = cropped.shape

    # Calculate scale factor preserving aspect ratio
    max_w = max(1, target_w - 2 * padding)
    max_h = max(1, target_h - 2 * padding)

    scale = min(max_w / crop_w, max_h / crop_h)
    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))

    # Resize using Pillow (NEAREST or BILINEAR + threshold)
    crop_pil = Image.fromarray(cropped)
    resized_pil = crop_pil.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    resized_arr = np.array(resized_pil, dtype=np.uint8)
    resized_binary = np.where(resized_arr > 100, 255, 0).astype(np.uint8)

    # Center inside target canvas
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    start_x = (target_w - new_w) // 2
    start_y = (target_h - new_h) // 2
    canvas[start_y : start_y + new_h, start_x : start_x + new_w] = resized_binary

    return canvas


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
        default_f = find_default_font()
        font_paths = [default_f] if default_f else []

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
