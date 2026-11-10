
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.core.base import BaseOCREngine


class TemplateMatchingOCR(BaseOCREngine):
    """Classical OCR engine based on image template matching."""

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
        """Recognize text from a PIL image or NumPy array."""

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

        except (ValueError, TypeError, OSError, cv2.error) as error:
            print(f"[OCR] Recognition error: {error}")
            return f"[{self.name}: Recognition error - {error}]"

    # ---------------------------------------------------------
    # TEMPLATE LOADING
    # ---------------------------------------------------------

    def _load_templates(self) -> None:
        """Load character templates using index.json metadata."""

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

            # Prevent paths outside the templates directory.
            try:
                template_path.resolve().relative_to(
                    self.templates_dir.resolve()
                )
            except ValueError:
                continue

            if not template_path.is_file():
                continue

            try:
                with Image.open(template_path) as image:
                    gray = np.asarray(
                        image.convert("L"),
                        dtype=np.uint8,
                    )

                binary = self._binarize(gray)

                if binary is None:
                    continue

                normalized = self._normalize_character(binary)

                if not np.any(normalized):
                    continue

                self.templates.append(
                    (character, normalized)
                )

            except (OSError, ValueError, cv2.error):
                continue

        print(f"[OCR] Successfully loaded {len(self.templates)} templates.")

    # ---------------------------------------------------------
    # IMAGE PREPROCESSING
    # ---------------------------------------------------------

    def _preprocess_image(
        self,
        image: Any,
    ) -> np.ndarray | None:
        """Convert input image into a binary foreground mask."""

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

        if not isinstance(image, np.ndarray):
            return None

        if image.size == 0:
            return None

        if image.ndim == 3:

            if image.shape[2] == 4:
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_RGBA2GRAY,
                )

            elif image.shape[2] == 3:
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_RGB2GRAY,
                )

            else:
                return None

        elif image.ndim != 2:
            return None

        if image.dtype != np.uint8:
            image = np.nan_to_num(image)
            image = np.clip(image, 0, 255).astype(np.uint8)

        if min(image.shape) < 2:
            return None

        return self._binarize(image)

    @staticmethod
    def _binarize(image: np.ndarray) -> np.ndarray | None:
        """Create a binary mask with white foreground."""

        if image.size == 0:
            return None

        blurred = cv2.GaussianBlur(
            image,
            (3, 3),
            0,
        )

        _, binary = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        border = np.concatenate(
            (
                binary[0, :],
                binary[-1, :],
                binary[:, 0],
                binary[:, -1],
            )
        )

        # Text should always be white.
        if np.median(border) > 127:
            binary = cv2.bitwise_not(binary)

        return binary

    # ---------------------------------------------------------
    # SEGMENTATION
    # ---------------------------------------------------------

    @staticmethod
    def _find_segments(
        projection: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Find continuous non-empty regions in a projection."""

        active = projection > 0

        segments = []
        start = None

        for index, value in enumerate(active):

            if value and start is None:
                start = index

            elif not value and start is not None:
                segments.append((start, index))
                start = None

        if start is not None:
            segments.append((start, len(active)))

        return segments

    def _segment_lines(
        self,
        image: np.ndarray,
    ) -> list[np.ndarray]:
        """Detect horizontal text lines."""

        # Join small vertical gaps, for example dots above i and j.
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, 5),
        )

        connected = cv2.morphologyEx(
            image,
            cv2.MORPH_CLOSE,
            kernel,
        )

        projection = np.sum(
            connected > 0,
            axis=1,
        )

        segments = self._find_segments(projection)

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
        """Detect individual characters using vertical projection."""

        projection = np.sum(
            line > 0,
            axis=0,
        )

        segments = self._find_segments(projection)

        characters = []

        for start, end in segments:

            character = line[:, start:end]

            rows = np.where(
                np.any(character > 0, axis=1)
            )[0]

            if rows.size == 0:
                continue

            top = int(rows[0])
            bottom = int(rows[-1]) + 1

            cropped = character[top:bottom, :]

            height, width = cropped.shape

            if (
                width < self.MIN_CHARACTER_WIDTH
                or height < self.MIN_CHARACTER_HEIGHT
            ):
                continue

            characters.append(
                (cropped, start, end)
            )

        return characters

    # ---------------------------------------------------------
    # CHARACTER NORMALIZATION
    # ---------------------------------------------------------

    def _normalize_character(
        self,
        character: np.ndarray,
    ) -> np.ndarray:
        """Resize and center character while preserving proportions."""

        canvas = np.zeros(
            (
                self.TEMPLATE_SIZE,
                self.TEMPLATE_SIZE,
            ),
            dtype=np.uint8,
        )

        rows, columns = np.where(character > 0)

        if rows.size == 0:
            return canvas

        cropped = character[
            rows.min():rows.max() + 1,
            columns.min():columns.max() + 1,
        ]

        height, width = cropped.shape

        if height == 0 or width == 0:
            return canvas

        scale = min(
            self.CHARACTER_SIZE / width,
            self.CHARACTER_SIZE / height,
        )

        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))

        resized = cv2.resize(
            cropped,
            (new_width, new_height),
            interpolation=(
                cv2.INTER_AREA
                if scale < 1
                else cv2.INTER_LINEAR
            ),
        )

        resized = np.where(
            resized > 127,
            255,
            0,
        ).astype(np.uint8)

        x_offset = (self.TEMPLATE_SIZE - new_width) // 2
        y_offset = (self.TEMPLATE_SIZE - new_height) // 2

        canvas[
            y_offset:y_offset + new_height,
            x_offset:x_offset + new_width,
        ] = resized

        return canvas

    # ---------------------------------------------------------
    # TEMPLATE COMPARISON
    # ---------------------------------------------------------

    @staticmethod
    def _calculate_similarity(
        character: np.ndarray,
        template: np.ndarray,
    ) -> float:
        """Calculate combined correlation and Dice similarity."""

        first = character.astype(np.uint8)
        second = template.astype(np.uint8)

        # Normalized cross-correlation.
        correlation = cv2.matchTemplate(
            first,
            second,
            cv2.TM_CCOEFF_NORMED,
        )

        correlation_score = float(correlation[0, 0])

        if not np.isfinite(correlation_score):
            correlation_score = 0.0

        correlation_score = (
            correlation_score + 1.0
        ) / 2.0

        # Dice similarity.
        first_mask = first > 0
        second_mask = second > 0

        intersection = np.count_nonzero(
            first_mask & second_mask
        )

        total = (
            np.count_nonzero(first_mask)
            + np.count_nonzero(second_mask)
        )

        dice_score = (
            2.0 * intersection / total
            if total > 0
            else 0.0
        )

        # Combined score.
        return (
            0.65 * correlation_score
            + 0.35 * dice_score
        )

    @staticmethod
    def _format_char_for_console(char: str) -> str:
        """Safely encode character for console printing across all OS encodings."""
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
        """Find the most similar character template and log comparisons to console."""

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
        """Recognize characters and spaces in one text line."""

        characters = self._segment_characters(line)

        if not characters:
            return ""

        widths = [
            end - start
            for _, start, end in characters
        ]

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

            # Detect word separation.
            if previous_end is not None:

                gap = start - previous_end

                if gap >= space_threshold:
                    result.append(" ")

            char_counter[0] += 1
            char_h, char_w = character.shape[:2]

            normalized = self._normalize_character(
                character
            )

            recognized = self._match_template(
                normalized,
                char_index=char_counter[0],
                original_dim=(char_w, char_h),
            )

            result.append(recognized)

            previous_end = end

        return "".join(result).strip()
