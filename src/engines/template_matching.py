"""Classical Optical Character Recognition (OCR) engine based on Template Matching.

This engine utilizes the project's ImagePreprocessor pipeline for image binarization,
line segmentation, character isolation, and glyph normalization, combined with
normalized cross-correlation and Dice similarity against reference templates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.core.base import BaseOCREngine
from src.core.preprocessor import ImagePreprocessor


class TemplateMatchingOCR(BaseOCREngine):
    """Classical OCR engine based on reference template matching."""

    TEMPLATE_SIZE = 32
    CHARACTER_SIZE = 24

    MIN_CHARACTER_WIDTH = 1
    MIN_CHARACTER_HEIGHT = 2

    SPACE_WIDTH_FACTOR = 0.65
    MIN_SPACE_WIDTH = 3

    MATCH_THRESHOLD = 0.35

    def __init__(
        self,
        templates_dir: Path | None = None,
        match_threshold: float | None = None,
    ) -> None:
        """Initialize TemplateMatchingOCR engine and load reference templates.

        Args:
            templates_dir: Optional custom path to templates folder containing index.json.
            match_threshold: Optional confidence threshold below which characters are marked as '?'.
        """
        super().__init__(name="Template Matching")

        if match_threshold is not None:
            self.MATCH_THRESHOLD = float(match_threshold)

        project_root = Path(__file__).resolve().parents[2]

        if templates_dir is not None:
            self.templates_dir = Path(templates_dir)
        else:
            try:
                from src.utils.helpers import get_resource_path
                self.templates_dir = get_resource_path("data/templates")
            except Exception:
                self.templates_dir = project_root / "data" / "templates"

        self.templates: list[tuple[str, np.ndarray]] = []

        self._load_templates()

        print(f"[OCR] Engine initialized: {self.name}")
        print(f"[OCR] Templates directory: {self.templates_dir}")
        print(f"[OCR] Loaded templates: {len(self.templates)}")

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------

    def recognize(
        self,
        image: Any,
        image_name: str | None = None,
    ) -> str:
        """Recognize text from an image (PIL Image, NumPy array, or file path).

        Args:
            image: Input image (PIL.Image, numpy.ndarray, or file path str/Path).
            image_name: Optional label or filename for console logging.

        Returns:
            Recognized text string with reconstructed words and lines.
        """
        if image is None:
            return f"[{self.name}: No image provided]"

        if not self.templates:
            print("[OCR] ERROR: No templates loaded.")
            return f"[{self.name}: Templates not found]"

        if image_name is None and isinstance(image, (str, Path)):
            image_name = Path(image).name

        print("\n" + "=" * 50)
        print("TEMPLATE MATCHING OCR")
        print("==================================================")
        print(f"\nZaładowano szablonów: {len(self.templates)}\n")
        if image_name:
            print(f"Obraz: {image_name}\n")

        try:
            binary_image = self._preprocess_image(image)

            if binary_image is None or not np.any(binary_image):
                print("[OCR] ERROR: Invalid input image or no text detected.")
                return f"[{self.name}: No text detected]"

            lines = self._segment_lines(binary_image)

            if not lines:
                return f"[{self.name}: No text detected]"

            recognized_lines = []
            char_counter = [0]

            for line in lines:
                recognized = self._recognize_line(line, char_counter=char_counter)

                if recognized:
                    recognized_lines.append(recognized)

            if not recognized_lines:
                return f"[{self.name}: No text detected]"

            result = "\n".join(recognized_lines)

            print("==================================================")
            print(f"ROZPOZNANY TEKST: {result}")
            print("==================================================\n")

            return result

        except (ValueError, TypeError, OSError) as error:
            print(f"[OCR] Recognition error: {error}")
            return f"[{self.name}: Recognition error - {error}]"

    # ---------------------------------------------------------
    # TEMPLATE LOADING
    # ---------------------------------------------------------

    def _load_templates(self) -> None:
        """Load and validate character templates using index.json metadata."""
        index_path = self.templates_dir / "index.json"

        if not index_path.is_file():
            print(f"[OCR] ERROR: Missing file: {index_path}")
            return

        try:
            with index_path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            print(f"[OCR] ERROR: Cannot read index.json: {error}")
            return

        if not isinstance(metadata, list):
            print("[OCR] ERROR: Invalid template metadata.")
            return

        for entry in metadata:
            if not isinstance(entry, dict):
                continue

            character = entry.get("char")
            filename = entry.get("filename")

            if not isinstance(character, str) or len(character) != 1:
                continue

            if not isinstance(filename, str) or not filename:
                continue

            template_path = self.templates_dir / filename

            # Guard against directory traversal attacks outside templates directory
            try:
                template_path.resolve().relative_to(self.templates_dir.resolve())
            except ValueError:
                continue

            if not template_path.is_file():
                continue

            try:
                with Image.open(template_path) as img:
                    # Validate image dimensions and convert to grayscale numpy array
                    if img.size[0] < 2 or img.size[1] < 2:
                        continue
                    gray = np.asarray(img.convert("L"), dtype=np.uint8)

                # Binarize and invert to foreground mask using ImagePreprocessor
                binary = self._binarize(gray)
                if binary is None:
                    continue

                normalized = self._normalize_character(binary)
                if not np.any(normalized):
                    continue

                self.templates.append((character, normalized))

            except (OSError, ValueError):
                continue

        print(f"Załadowano szablonów: {len(self.templates)}")
        print(f"[OCR] Successfully loaded {len(self.templates)} templates.")

    # ---------------------------------------------------------
    # IMAGE PREPROCESSING (DELEGATED TO ImagePreprocessor)
    # ---------------------------------------------------------

    def _preprocess_image(self, image: Any) -> np.ndarray | None:
        """Convert input image into a normalized binary foreground mask using ImagePreprocessor.

        Args:
            image: Input image (PIL.Image, numpy.ndarray, or file path).

        Returns:
            2D numpy uint8 array where text pixels=255 and background=0, or None if invalid.
        """
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.is_file():
                return None
            try:
                with Image.open(path) as img:
                    image = np.asarray(img.convert("RGB"))
            except (OSError, ValueError):
                return None

        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB"))

        if not isinstance(image, np.ndarray) or image.size == 0:
            return None

        # Convert to grayscale using ImagePreprocessor
        gray = ImagePreprocessor.to_grayscale(image)
        if not isinstance(gray, np.ndarray) or gray.ndim != 2 or min(gray.shape) < 2:
            return None

        if gray.dtype != np.uint8:
            gray = np.nan_to_num(gray)
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        # Apply Otsu's binarization using ImagePreprocessor
        return self._binarize(gray)

    @staticmethod
    def _binarize(image: np.ndarray) -> np.ndarray | None:
        """Create a binary mask with white text (255) and black background (0) using ImagePreprocessor."""
        if image is None or image.size == 0:
            return None

        # Apply Otsu thresholding from ImagePreprocessor
        binary = ImagePreprocessor.otsu(image)

        # Normalize polarity so text is 255 and background is 0
        return ImagePreprocessor.invert_to_foreground_mask(binary)

    # ---------------------------------------------------------
    # SEGMENTATION (DELEGATED TO ImagePreprocessor)
    # ---------------------------------------------------------

    @staticmethod
    def _find_segments(projection: np.ndarray) -> list[tuple[int, int]]:
        """Find continuous active regions in a projection profile using ImagePreprocessor."""
        return ImagePreprocessor.find_projection_profile_segments(
            projection,
            threshold=0,
            min_size=1,
            min_gap=1,
        )

    def _segment_lines(self, image: np.ndarray) -> list[np.ndarray]:
        """Detect horizontal text lines using horizontal projection profile segments."""
        mask = ImagePreprocessor.invert_to_foreground_mask(image)
        h_profile = np.sum(mask > 0, axis=1)

        segments = ImagePreprocessor.find_projection_profile_segments(
            h_profile,
            threshold=0,
            min_size=self.MIN_CHARACTER_HEIGHT,
            min_gap=1,
        )

        lines = []
        for start, end in segments:
            if end <= start:
                continue
            line = image[start:end, :]
            if np.any(line):
                lines.append(line)

        return lines

    def _segment_characters(
        self,
        line: np.ndarray,
    ) -> list[tuple[np.ndarray, int, int]]:
        """Detect individual character glyphs using ImagePreprocessor projection profile segments."""
        line_mask = ImagePreprocessor.invert_to_foreground_mask(line)
        v_profile = np.sum(line_mask > 0, axis=0)

        segments = ImagePreprocessor.find_projection_profile_segments(
            v_profile,
            threshold=0,
            min_size=self.MIN_CHARACTER_WIDTH,
            min_gap=1,
        )

        characters = []
        for start, end in segments:
            character_slice = line_mask[:, start:end]
            rows = np.where(np.any(character_slice > 0, axis=1))[0]
            if rows.size == 0:
                continue

            top = int(rows[0])
            bottom = int(rows[-1]) + 1
            cropped = character_slice[top:bottom, :]

            height, width = cropped.shape
            if width < self.MIN_CHARACTER_WIDTH or height < self.MIN_CHARACTER_HEIGHT:
                continue

            characters.append((cropped, int(start), int(end)))

        return characters

    # ---------------------------------------------------------
    # CHARACTER NORMALIZATION (DELEGATED TO ImagePreprocessor)
    # ---------------------------------------------------------

    def _normalize_character(self, character: np.ndarray) -> np.ndarray:
        """Resize and center character onto a fixed target canvas using ImagePreprocessor."""
        return ImagePreprocessor.normalize_character(
            character,
            target_size=(self.TEMPLATE_SIZE, self.TEMPLATE_SIZE),
            padding=2,
        )

    # ---------------------------------------------------------
    # TEMPLATE COMPARISON & SIMILARITY METRICS
    # ---------------------------------------------------------

    @staticmethod
    def _calculate_similarity(
        character: np.ndarray,
        template: np.ndarray,
    ) -> float:
        """Calculate combined normalized cross-correlation and Dice similarity using pure NumPy."""
        first = character.astype(np.float64)
        second = template.astype(np.float64)

        # Normalized cross-correlation (Pearson correlation coefficient)
        first_centered = first - np.mean(first)
        second_centered = second - np.mean(second)
        denominator = np.sqrt(np.sum(first_centered ** 2) * np.sum(second_centered ** 2))

        if denominator > 0:
            correlation_score = float(np.sum(first_centered * second_centered) / denominator)
        else:
            correlation_score = 0.0

        if not np.isfinite(correlation_score):
            correlation_score = 0.0

        correlation_score = (correlation_score + 1.0) / 2.0

        # Dice similarity
        first_mask = first > 0
        second_mask = second > 0
        intersection = np.count_nonzero(first_mask & second_mask)
        total = np.count_nonzero(first_mask) + np.count_nonzero(second_mask)

        dice_score = (2.0 * intersection / total) if total > 0 else 0.0

        # Weighted combination: 65% NCC + 35% Dice
        return 0.65 * correlation_score + 0.35 * dice_score

    @staticmethod
    def _format_char_for_console(char: str) -> str:
        """Safely encode character for console printing across diverse OS terminal encodings."""
        try:
            encoding = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
            char.encode(encoding)
            return char
        except (UnicodeEncodeError, AttributeError):
            return repr(char)

    def _match_template(
        self,
        character: np.ndarray,
        char_index: int | None = None,
        original_dim: tuple[int, int] | None = None,
    ) -> str:
        """Find the best matching character template and log detailed comparisons to console."""
        if not self.templates:
            return "?"

        if char_index is not None:
            print(f"ZNAK {char_index}:")
            if original_dim is not None:
                w, h = original_dim
                print(f"  Wymiary fragmentu: {w}x{h} px")
        elif original_dim is not None:
            w, h = original_dim
            print("ZNAK:")
            print(f"  Wymiary fragmentu: {w}x{h} px")

        best_character = "?"
        best_score = -1.0

        for template_character, template in self.templates:
            score = self._calculate_similarity(character, template)
            pct = score * 100.0

            safe_char = self._format_char_for_console(template_character)
            print(f"  Porównanie z {safe_char}: {pct:.2f}%")

            if score > best_score:
                best_score = score
                best_character = template_character

        if best_score < self.MATCH_THRESHOLD:
            status = f"odrzucono (poniżej progu {self.MATCH_THRESHOLD * 100:.1f}%)"
            accepted_char = "?"
        else:
            status = "zaakceptowano"
            accepted_char = best_character

        safe_best = self._format_char_for_console(best_character)
        print(f"\n  Najlepsze dopasowanie: {safe_best}")
        print(f"  Podobieństwo: {best_score * 100.0:.2f}%")
        print(f"  Status: {status}\n")

        return accepted_char

    # ---------------------------------------------------------
    # TEXT RECOGNITION
    # ---------------------------------------------------------

    def _recognize_line(
        self,
        line: np.ndarray,
        char_counter: list[int] | None = None,
    ) -> str:
        """Recognize characters and space intervals in one text line."""
        characters = self._segment_characters(line)

        if not characters:
            return ""

        widths = [end - start for _, start, end in characters]
        median_width = float(np.median(widths))

        space_threshold = max(
            self.MIN_SPACE_WIDTH,
            median_width * self.SPACE_WIDTH_FACTOR,
        )

        result = []
        previous_end = None

        if char_counter is None:
            char_counter = [0]

        for character, start, end in characters:
            # Detect word spacing
            if previous_end is not None:
                gap = start - previous_end
                if gap >= space_threshold:
                    result.append(" ")

            char_counter[0] += 1
            char_h, char_w = character.shape[:2]

            normalized = self._normalize_character(character)
            recognized = self._match_template(
                normalized,
                char_index=char_counter[0],
                original_dim=(char_w, char_h),
            )

            result.append(recognized)
            previous_end = end

        return "".join(result).strip()
