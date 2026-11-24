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

try:
    import numba

    @numba.njit(fastmath=True)
    def _calculate_similarity_numba(first: np.ndarray, second: np.ndarray) -> float:
        h, w = first.shape
        total_pixels = h * w

        sum_f = 0.0
        sum_s = 0.0
        for r in range(h):
            for c in range(w):
                sum_f += first[r, c]
                sum_s += second[r, c]
        mean_f = sum_f / total_pixels
        mean_s = sum_s / total_pixels

        cross_sum = 0.0
        var_f = 0.0
        var_s = 0.0
        intersection = 0
        total = 0

        for r in range(h):
            for c in range(w):
                f_val = first[r, c]
                s_val = second[r, c]
                df = f_val - mean_f
                ds = s_val - mean_s
                cross_sum += df * ds
                var_f += df * df
                var_s += ds * ds

                f_mask = 1 if f_val > 0.0 else 0
                s_mask = 1 if s_val > 0.0 else 0
                if f_mask != 0 and s_mask != 0:
                    intersection += 1
                total += f_mask + s_mask

        denom = np.sqrt(var_f * var_s)
        if denom > 0.0:
            corr = cross_sum / denom
        else:
            corr = 0.0

        if np.isnan(corr) or np.isinf(corr):
            corr = 0.0
        corr = (corr + 1.0) / 2.0

        dice = (2.0 * intersection / total) if total > 0 else 0.0
        return 0.65 * corr + 0.35 * dice

    @numba.njit(fastmath=True, parallel=True)
    def _batch_calculate_similarity_numba(
        character: np.ndarray,
        templates: np.ndarray,
    ) -> np.ndarray:
        N, H, W = templates.shape
        scores = np.empty(N, dtype=np.float64)
        total_pixels = H * W

        sum_c = 0.0
        for r in range(H):
            for c in range(W):
                sum_c += character[r, c]
        mean_c = sum_c / total_pixels

        var_c = 0.0
        total_c_nonzero = 0
        for r in range(H):
            for c in range(W):
                dc = character[r, c] - mean_c
                var_c += dc * dc
                if character[r, c] > 0.0:
                    total_c_nonzero += 1

        for i in numba.prange(N):
            tpl = templates[i]
            sum_t = 0.0
            for r in range(H):
                for c in range(W):
                    sum_t += tpl[r, c]
            mean_t = sum_t / total_pixels

            cross_sum = 0.0
            var_t = 0.0
            intersection = 0
            total_t_nonzero = 0
            for r in range(H):
                for c in range(W):
                    c_val = character[r, c]
                    t_val = tpl[r, c]
                    dc = c_val - mean_c
                    dt = t_val - mean_t
                    cross_sum += dc * dt
                    var_t += dt * dt
                    if (c_val > 0.0) and (t_val > 0.0):
                        intersection += 1
                    if t_val > 0.0:
                        total_t_nonzero += 1

            denom = np.sqrt(var_c * var_t)
            if denom > 0.0:
                corr = cross_sum / denom
            else:
                corr = 0.0

            if np.isnan(corr) or np.isinf(corr):
                corr = 0.0
            corr = (corr + 1.0) / 2.0

            tot = total_c_nonzero + total_t_nonzero
            dice = (2.0 * intersection / tot) if tot > 0 else 0.0
            scores[i] = 0.65 * corr + 0.35 * dice

        return scores

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

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
        self._template_matrix: np.ndarray | None = None
        self._template_chars: list[str] = []

        self._load_templates()

        if self.templates:
            self._template_chars = [c for c, _ in self.templates]
            self._template_matrix = np.stack(
                [t.astype(np.float64) for _, t in self.templates]
            )

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

        if image_name:
            print(f"Obraz: {image_name}")

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

            return result

        except (ValueError, TypeError, OSError) as error:
            print(f"[OCR] Recognition error: {error}")
            return f"[{self.name}: Recognition error - {error}]"

    # ---------------------------------------------------------
    # TEMPLATE LOADING
    # ---------------------------------------------------------

    def _load_templates(self) -> None:
        """Load and validate character templates using index.json metadata (or cache if available)."""
        cache_path = self.templates_dir / "templates_cache.npz"
        index_path = self.templates_dir / "index.json"

        if cache_path.is_file():
            try:
                cache_mtime = cache_path.stat().st_mtime
                index_mtime = index_path.stat().st_mtime if index_path.is_file() else 0
                if cache_mtime >= index_mtime:
                    data = np.load(cache_path)
                    chars = [str(c) for c in data["chars"]]
                    tpls = data["templates"]
                    self.templates = [(chars[i], tpls[i]) for i in range(len(chars))]
                    return
            except Exception:
                self.templates = []

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

        print(f"[OCR] Successfully loaded {len(self.templates)} templates.")

        if self.templates and index_path.is_file():
            try:
                chars_arr = np.array([c for c, _ in self.templates])
                tpls_arr = np.stack([t for _, t in self.templates])
                np.savez_compressed(cache_path, chars=chars_arr, templates=tpls_arr)
            except Exception:
                pass

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
        """Detect individual character glyphs using connected components and projection fallback."""
        line_mask = ImagePreprocessor.invert_to_foreground_mask(line)
        if not np.any(line_mask > 0):
            return []

        # First attempt robust connected-components segmentation with touching-glyph splitting
        try:
            char_dicts = ImagePreprocessor.segment_characters_and_spaces(
                line_mask,
                space_ratio_threshold=0.38,
                min_char_width=self.MIN_CHARACTER_WIDTH,
                min_char_height=self.MIN_CHARACTER_HEIGHT,
            )
            if char_dicts:
                return [
                    (c["crop"], int(c["bbox"][0]), int(c["bbox"][0] + c["bbox"][2]))
                    for c in char_dicts
                ]
        except Exception:
            pass

        # Fallback to projection profile segmentation
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
        """Calculate combined normalized cross-correlation and Dice similarity."""
        first = character.astype(np.float64)
        second = template.astype(np.float64)

        if _NUMBA_AVAILABLE:
            return float(_calculate_similarity_numba(first, second))

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
        line_context: dict[str, Any] | None = None,
    ) -> str:
        """Find the best matching character template using feature-weighted similarity."""
        if not self.templates:
            return "?"

        if (
            _NUMBA_AVAILABLE
            and self._template_matrix is not None
            and len(self._template_matrix) == len(self.templates)
        ):
            scores = _batch_calculate_similarity_numba(
                character.astype(np.float64),
                self._template_matrix,
            )
            char_scores: dict[str, float] = {}
            for i, t_char in enumerate(self._template_chars):
                s = float(scores[i])
                if s > char_scores.get(t_char, -1.0):
                    char_scores[t_char] = s
        else:
            char_scores = {}
            for t_char, template in self.templates:
                s = self._calculate_similarity(character, template)
                if s > char_scores.get(t_char, -1.0):
                    char_scores[t_char] = s

        if not char_scores:
            return "?"

        # Dampen rare symbols so they do not overshadow natural letters
        for sym in "{}[]«»°~^|":
            if sym in char_scores:
                char_scores[sym] -= 0.04

        best_score = max(char_scores.values())
        if best_score < self.MATCH_THRESHOLD:
            return "?"

        accepted_char = max(char_scores, key=char_scores.get)

        # Apply typographic height and baseline disambiguation if context is available
        if line_context and original_dim:
            cw, ch = original_dim
            line_h = int(line_context.get("line_h", 20))
            med_h = float(line_context.get("med_h", 14.0))
            by = int(line_context.get("by", 0))
            has_descenders = bool(line_context.get("has_descenders", False))

            is_cap_height = (ch >= med_h * 1.12) or (by <= int(line_h * 0.15) and ch >= int(line_h * 0.85))
            is_x_height = (ch <= int(med_h * 1.02)) and (by >= int(line_h * 0.18))

            if is_cap_height:
                if accepted_char in ("o", "s", "c", "v", "w", "z", "x"):
                    accepted_char = accepted_char.upper()
                elif accepted_char == "r":
                    accepted_char = "E"
                elif accepted_char == "p" and not has_descenders:
                    accepted_char = "P"
            elif is_x_height:
                if accepted_char in ("O", "S", "C", "V", "W", "Z", "X"):
                    accepted_char = accepted_char.lower()
                elif accepted_char == "P":
                    accepted_char = "p"

            if accepted_char in ("'", "’", "`") and ch >= int(med_h * 0.75):
                accepted_char = "1"

        return accepted_char

    def predict_glyph(self, glyph_crop: np.ndarray) -> str:
        """Predicts the character for a single glyph crop using template matching.

        Args:
            glyph_crop: 2D binary numpy array representing character glyph.

        Returns:
            Predicted character string.
        """
        if not self.templates:
            raise RuntimeError("Template matching templates not loaded.")

        if glyph_crop.shape != (self.TEMPLATE_SIZE, self.TEMPLATE_SIZE):
            glyph = self._normalize_character(glyph_crop)
        else:
            glyph = glyph_crop.astype(np.uint8)

        return self._match_template(glyph)

    # ---------------------------------------------------------
    # TEXT RECOGNITION
    # ---------------------------------------------------------

    def _recognize_line(
        self,
        line: np.ndarray,
        char_counter: list[int] | None = None,
    ) -> str:
        """Recognize characters and space intervals in one text line."""
        line_mask = ImagePreprocessor.invert_to_foreground_mask(line)
        if not np.any(line_mask > 0):
            return ""

        line_h = line_mask.shape[0]

        # Use robust character segmentation
        char_dicts = []
        try:
            char_dicts = ImagePreprocessor.segment_characters_and_spaces(
                line_mask,
                space_ratio_threshold=0.38,
                min_char_width=self.MIN_CHARACTER_WIDTH,
                min_char_height=self.MIN_CHARACTER_HEIGHT,
            )
        except Exception:
            pass

        if not char_dicts:
            legacy_chars = self._segment_characters(line)
            if not legacy_chars:
                return ""
            char_dicts = [
                {
                    "crop": c,
                    "bbox": (start, 0, end - start, line_h),
                    "is_space_before": False,
                }
                for c, start, end in legacy_chars
            ]

        # Calculate line height statistics
        char_hs = [c["bbox"][3] for c in char_dicts if c["bbox"][3] >= 5]
        med_h = float(np.median(char_hs)) if char_hs else float(line_h * 0.7)
        char_ws = [c["bbox"][2] for c in char_dicts if c["bbox"][2] >= 3]
        med_w = float(np.median(char_ws)) if char_ws else 12.0

        # Detect presence of descenders across line
        has_descenders = any((c["bbox"][1] + c["bbox"][3]) > int(med_h * 1.3) for c in char_dicts)

        # Dynamic space threshold based on inter-character gaps
        gaps = [
            char_dicts[k + 1]["bbox"][0] - (char_dicts[k]["bbox"][0] + char_dicts[k]["bbox"][2])
            for k in range(len(char_dicts) - 1)
        ]
        pos_gaps = [g for g in gaps if g > 0]
        med_gap = float(np.median(pos_gaps)) if pos_gaps else 2.0
        space_threshold = max(4.0, min(med_w * 0.45, med_gap * 2.2))

        result: list[str] = []
        prev_end: int | None = None

        if char_counter is None:
            char_counter = [0]

        for char_info in char_dicts:
            crop = char_info["crop"]
            bx, by, bw, bh = char_info["bbox"]

            if prev_end is not None:
                gap = bx - prev_end
                if gap >= space_threshold or char_info.get("is_space_before", False):
                    result.append(" ")

            char_counter[0] += 1
            normalized = self._normalize_character(crop)
            line_ctx = {
                "line_h": line_h,
                "med_h": med_h,
                "by": by,
                "has_descenders": has_descenders,
            }

            recognized = self._match_template(
                normalized,
                char_index=char_counter[0],
                original_dim=(bw, bh),
                line_context=line_ctx,
            )

            result.append(recognized)
            prev_end = bx + bw

        return "".join(result).strip()
