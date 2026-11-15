from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier

from src.core.base import BaseOCREngine
from src.core.preprocessor import ImagePreprocessor
from src.utils.helpers import get_resource_path

logger = logging.getLogger(__name__)


class HogKnnOCR(BaseOCREngine):
    """OCR engine based on Histogram of Oriented Gradients (HOG) features and k-NN classifier."""

    DEFAULT_ORIENTATIONS: int = 9
    DEFAULT_PIXELS_PER_CELL: tuple[int, int] = (8, 8)
    DEFAULT_CELLS_PER_BLOCK: tuple[int, int] = (2, 2)
    DEFAULT_NORMALIZED_SIZE: tuple[int, int] = (32, 32)
    DEFAULT_BLOCK_NORM: str = "L2-Hys"

    def __init__(
        self,
        model_path: Path | str | None = None,
        n_neighbors: int = 1,
    ) -> None:
        """Initializes the HOG + k-NN OCR Engine.

        Args:
            model_path: Optional path to a serialized trained model file (.pkl).
                Defaults to data/models/hog_knn_model.pkl.
            n_neighbors: Number of neighbors for the k-NN classifier.
        """
        super().__init__(name="HOG + k-NN")

        self.model_path = Path(model_path) if model_path else None
        self.n_neighbors = n_neighbors

        self.classifier: KNeighborsClassifier | None = None
        self.is_trained: bool = False

        self._initialize_model()

    @property
    def is_ready(self) -> bool:
        """Returns True if the k-NN classifier is loaded and ready for inference."""
        return self.is_trained and self.classifier is not None

    def _initialize_model(self) -> None:
        """Loads a model from model_path or the default model artifact location."""
        if self.model_path and self.model_path.exists():
            try:
                self.load(self.model_path)
                return
            except Exception as e:
                logger.warning("Failed to load model from %s: %s.", self.model_path, e)

        default_model = get_resource_path("data/models/hog_knn_model.pkl")
        if default_model.exists():
            try:
                self.load(default_model)
                return
            except Exception as e:
                logger.warning("Failed to load default model from %s: %s.", default_model, e)
        else:
            logger.info("No pre-trained model found at %s. Run scripts/train_hog_knn.py to train.", default_model)

    def extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """Extracts HOG descriptor feature vector from a single normalized glyph image.

        Args:
            image: 2D uint8 numpy array of shape (32, 32).

        Returns:
            1D float32 numpy array representing the HOG feature vector.
        """
        if len(image.shape) != 2:
            image = ImagePreprocessor.to_grayscale(image)

        if image.shape != self.DEFAULT_NORMALIZED_SIZE:
            image = cv2.resize(image, self.DEFAULT_NORMALIZED_SIZE, interpolation=cv2.INTER_AREA)

        features = hog(
            image,
            orientations=self.DEFAULT_ORIENTATIONS,
            pixels_per_cell=self.DEFAULT_PIXELS_PER_CELL,
            cells_per_block=self.DEFAULT_CELLS_PER_BLOCK,
            block_norm=self.DEFAULT_BLOCK_NORM,
        )
        return np.asarray(features, dtype=np.float32)

    def save(self, file_path: Path | str) -> None:
        """Serializes and saves the trained classifier to disk.

        Args:
            file_path: Output path for the pickled model.
        """
        if not self.is_trained or self.classifier is None:
            raise RuntimeError("Cannot save model: classifier is not trained.")

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.classifier, f)

    def load(self, file_path: Path | str) -> None:
        """Loads a serialized classifier from disk.

        Args:
            file_path: Path to the pickled model.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")

        with open(path, "rb") as f:
            self.classifier = pickle.load(f)

        self.is_trained = True

    def predict_glyph(self, glyph_crop: np.ndarray) -> str:
        """Predicts the character for a single glyph crop.

        Args:
            glyph_crop: 2D binary numpy array of character glyph.

        Returns:
            Predicted character string.
        """
        if not self.is_trained or self.classifier is None:
            raise RuntimeError("HOG + k-NN model is not trained or loaded.")

        normalized = ImagePreprocessor.normalize_character(
            glyph_crop,
            target_size=self.DEFAULT_NORMALIZED_SIZE,
            is_foreground=True,
        )
        feat = self.extract_hog_features(normalized)
        prediction = self.classifier.predict([feat])
        return str(prediction[0])

    def recognize(self, image: Any) -> str:
        """Process the input image and return the recognized text.

        Pipeline:
        1. Preprocess: to_grayscale -> Otsu binarization -> invert_to_foreground_mask.
        2. Layout Analysis: segment into paragraphs, then into ordered text lines.
        3. Character & Space Segmentation: segment characters and detect whitespace gaps.
        4. Glyph Normalization: normalize each character onto standard 32x32 canvas.
        5. Feature Extraction & Classification: extract HOG descriptors and predict with k-NN.
        6. Text Reconstruction: assemble recognized characters, spaces, linebreaks, and paragraphs.

        Args:
            image: Input image as numpy ndarray or PIL.Image.

        Returns:
            Recognized text string.
        """
        if image is None:
            return "[HOG + k-NN: Recognition output in development]"

        # Support PIL Image conversion
        if isinstance(image, Image.Image):
            image = np.array(image)

        if not isinstance(image, np.ndarray):
            return "[HOG + k-NN: Invalid image format]"

        if not self.is_trained or self.classifier is None:
            return "[HOG + k-NN: Model not trained or initialized]"

        # Step 1: Preprocessing & Binarization
        gray = ImagePreprocessor.to_grayscale(image)
        binary = ImagePreprocessor.otsu(gray)
        fg_mask = ImagePreprocessor.invert_to_foreground_mask(binary)

        # Step 2: Hierarchical Layout Segmentation
        paragraphs = ImagePreprocessor.segment_paragraphs(fg_mask)
        if not paragraphs:
            return ""

        # Collect glyphs along with paragraph and line indices for batch prediction
        all_features: list[np.ndarray] = []
        char_locations: list[tuple[int, int, bool]] = []  # (para_idx, line_idx, is_space_before)
        char_meta: list[tuple[int, int, int, int, int, int, bool]] = []  # (cx, cy, cw, ch, line_h, line_median_h, has_hole)
        para_line_counts: dict[int, int] = {}

        def _has_closed_loop(glyph: np.ndarray) -> bool:
            contours, hierarchy = cv2.findContours(glyph.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is None:
                return False
            for h in hierarchy[0]:
                if h[2] != -1:
                    return True
            return False

        for p_idx, para in enumerate(paragraphs):
            para_crop = para["crop"]
            h_profile = np.sum(para_crop > 0, axis=1)
            raw_lines = ImagePreprocessor.find_projection_profile_segments(
                h_profile,
                threshold=0,
                min_size=5,
                min_gap=1,
            )

            para_line_counts[p_idx] = len(raw_lines)
            if not raw_lines:
                continue

            for l_idx, (y_s, y_e) in enumerate(raw_lines):
                line_slice = para_crop[y_s:y_e, :]
                chars = ImagePreprocessor.segment_characters_and_spaces(line_slice)
                if not chars:
                    continue

                line_h = line_slice.shape[0]
                char_hs = [c["bbox"][3] for c in chars if c["bbox"][3] > 6]
                sorted_hs = sorted(char_hs)
                line_median_h = int(sorted_hs[len(sorted_hs) // 3]) if sorted_hs else 14

                for char_dict in chars:
                    glyph_crop = char_dict["crop"]
                    normalized = ImagePreprocessor.normalize_character(
                        glyph_crop,
                        target_size=self.DEFAULT_NORMALIZED_SIZE,
                        is_foreground=True,
                    )
                    feat = self.extract_hog_features(normalized)
                    all_features.append(feat)
                    char_locations.append((p_idx, l_idx, char_dict.get("is_space_before", False)))
                    bx, by, bw, bh = char_dict["bbox"]
                    has_loop = _has_closed_loop(glyph_crop)
                    char_meta.append((
                        bx,
                        by,
                        bw,
                        bh,
                        line_h,
                        line_median_h,
                        has_loop,
                    ))

        if not all_features:
            return ""

        # Batch predict all characters at once with k-NN
        predictions = self.classifier.predict(np.asarray(all_features, dtype=np.float32))

        # Reconstruct text hierarchy with typographic case disambiguation and homoglyph correction
        structured_text: dict[int, dict[int, list[str]]] = {}
        for (p_idx, l_idx, is_space_before), char_pred, (cx, cy, cw, ch, lh, med_h, has_loop) in zip(
            char_locations, predictions, char_meta, strict=True
        ):
            pred_str = str(char_pred)

            # 1. Topological loop disambiguation for b vs h
            if pred_str == "b" and not has_loop:
                pred_str = "h"
            elif pred_str == "h" and has_loop:
                pred_str = "b"

            # 2. Case disambiguation for homoglyphs (C/c, S/s, V/v, W/w, Z/z, X/x, U/u)
            # If inside a word (not preceded by space/boundary), ambiguous letters are lowercase
            is_word_initial = is_space_before or (p_idx not in structured_text) or (l_idx not in structured_text[p_idx]) or (not structured_text[p_idx][l_idx])
            if pred_str in ("C", "S", "V", "W", "Z", "X", "U"):
                if not is_word_initial and ch <= int(med_h * 1.25):
                    pred_str = pred_str.lower()
                elif is_word_initial and ch <= int(med_h * 0.95) and cy >= int(lh * 0.2):
                    pred_str = pred_str.lower()
            elif pred_str in ("c", "s", "v", "w", "z", "x", "u"):
                if is_word_initial and (ch >= int(med_h * 1.1) or cy <= int(lh * 0.15)):
                    pred_str = pred_str.upper()

            # 3. p vs P (lowercase p has a descender reaching bottom of line)
            if pred_str == "P" and (cy + ch) >= int(lh * 0.85):
                pred_str = "p"
            elif pred_str == "p" and (cy + ch) <= int(lh * 0.78) and cy <= int(lh * 0.15):
                pred_str = "P"

            # 4. y vs Y (lowercase y has descender)
            if pred_str == "Y" and (cy + ch) >= int(lh * 0.85):
                pred_str = "y"

            # 5. O vs o vs 0
            if pred_str in ("O", "0") and ch <= int(med_h * 1.18) and cy >= int(lh * 0.12):
                pred_str = "o"
            elif pred_str == "o" and (ch >= int(med_h * 1.35) or cy <= int(lh * 0.05)):
                pred_str = "O"

            # 6. Punctuation disambiguation
            if pred_str in ("’", "'", "‘", "}"):
                if cy >= int(lh * 0.45):
                    pred_str = ","
                else:
                    pred_str = "'"
            elif pred_str in (":", ";") and ch <= 7 and cw <= 7 and (cy + ch) >= int(lh * 0.65):
                pred_str = "."
            elif pred_str == "+" and cw / max(1, ch) >= 1.4 and ch <= 8:
                pred_str = "-"

            if p_idx not in structured_text:
                structured_text[p_idx] = {}
            if l_idx not in structured_text[p_idx]:
                structured_text[p_idx][l_idx] = []

            if is_space_before:
                structured_text[p_idx][l_idx].append(" ")
            structured_text[p_idx][l_idx].append(pred_str)

        recognized_paragraphs: list[str] = []
        for p_idx in sorted(structured_text.keys()):
            para_lines = structured_text[p_idx]
            clean_lines = []
            for l_idx in sorted(para_lines.keys()):
                raw_line = "".join(para_lines[l_idx])
                # Post-processing: disambiguate digits embedded in alphabetic words
                # e.g. 'wa11' -> 'wall', 'b1ue' -> 'blue', 'ca1led' -> 'called', 'fr0m' -> 'from'
                chars_list = list(raw_line)
                for ci in range(len(chars_list)):
                    ch_cur = chars_list[ci]
                    prev_c = chars_list[ci - 1] if ci > 0 else " "
                    next_c = chars_list[ci + 1] if ci < len(chars_list) - 1 else " "

                    # 1 vs l in word context
                    if ch_cur == "1" and (prev_c.isalpha() or next_c.isalpha()):
                        chars_list[ci] = "l"
                    # I at end of lowercase word, e.g. 'stilI' -> 'still'
                    elif ch_cur == "I" and prev_c.islower() and (next_c == " " or not next_c.isalpha()):
                        chars_list[ci] = "l"
                    # 0 in word context
                    elif ch_cur == "0" and (prev_c.islower() or next_c.islower()):
                        chars_list[ci] = "o"

                clean_lines.append("".join(chars_list))

            if clean_lines:
                recognized_paragraphs.append("\n".join(clean_lines))

        return "\n\n".join(recognized_paragraphs)
