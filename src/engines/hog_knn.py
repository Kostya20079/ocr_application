from __future__ import annotations

import json
import logging
import pickle
import warnings
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
        templates_dir: Path | str | None = None,
        n_neighbors: int = 1,
        auto_train_on_templates: bool = True,
    ) -> None:
        """Initializes the HOG + k-NN OCR Engine.

        Args:
            model_path: Optional path to a serialized trained model file (.pkl).
            templates_dir: Optional path to the templates directory (defaults to data/templates).
            n_neighbors: Number of neighbors for the k-NN classifier.
            auto_train_on_templates: Whether to automatically train or fit from templates
                if no pre-trained model file exists at `model_path`.
        """
        super().__init__(name="HOG + k-NN")

        self.model_path = Path(model_path) if model_path else None
        self.templates_dir = Path(templates_dir) if templates_dir else get_resource_path("data/templates")
        self.n_neighbors = n_neighbors
        self.auto_train_on_templates = auto_train_on_templates

        self.classifier: KNeighborsClassifier | None = None
        self.is_trained: bool = False

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Loads a model from model_path, or trains on templates if auto_train_on_templates is True."""
        if self.model_path and self.model_path.exists():
            try:
                self.load(self.model_path)
                return
            except Exception as e:
                logger.warning("Failed to load model from %s: %s. Falling back to templates.", self.model_path, e)

        default_model = get_resource_path("data/models/hog_knn_model.pkl")
        if default_model.exists():
            try:
                self.load(default_model)
                return
            except Exception as e:
                logger.warning("Failed to load default model from %s: %s.", default_model, e)

        if self.auto_train_on_templates and self.templates_dir.exists():
            self.train_from_templates(self.templates_dir)

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

    def train(self, features: np.ndarray, labels: list[str] | np.ndarray) -> None:
        """Fits the k-NN classifier on provided feature vectors and label targets.

        Args:
            features: 2D numpy array of shape (N, feature_dim).
            labels: List or 1D array of character string labels of length N.
        """
        X = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels)

        if len(X) == 0:
            raise ValueError("Training dataset cannot be empty.")
        if len(X) != len(y):
            raise ValueError(f"Features length ({len(X)}) does not match labels length ({len(y)}).")

        self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric="euclidean")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            self.classifier.fit(X, y)

        self.is_trained = True

    def train_from_templates(self, templates_dir: Path | str | None = None) -> None:
        """Builds dataset and fits k-NN using template images and index.json.

        Args:
            templates_dir: Directory containing template PNG files and index.json.
        """
        t_dir = Path(templates_dir) if templates_dir else self.templates_dir
        index_path = t_dir / "index.json"

        if not index_path.exists():
            logger.warning("Templates index file not found at: %s", index_path)
            return

        with open(index_path, "r", encoding="utf-8") as f:
            metadata: list[dict[str, Any]] = json.load(f)

        features: list[np.ndarray] = []
        labels: list[str] = []

        for entry in metadata:
            filename = entry.get("filename")
            char = entry.get("char")
            if not filename or char is None:
                continue

            img_path = t_dir / filename
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            feat = self.extract_hog_features(img)
            features.append(feat)
            labels.append(char)

        if features:
            self.train(np.array(features, dtype=np.float32), labels)
            logger.info("Trained HOG + k-NN model on %d templates.", len(features))

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
        para_line_counts: dict[int, int] = {}

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

                for char_dict in chars:
                    glyph_crop = char_dict["crop"]
                    normalized = ImagePreprocessor.normalize_character(
                        glyph_crop,
                        target_size=self.DEFAULT_NORMALIZED_SIZE,
                    )
                    feat = self.extract_hog_features(normalized)
                    all_features.append(feat)
                    char_locations.append((p_idx, l_idx, char_dict.get("is_space_before", False)))

        if not all_features:
            return ""

        # Batch predict all characters at once with k-NN
        predictions = self.classifier.predict(np.asarray(all_features, dtype=np.float32))

        # Reconstruct text hierarchy
        structured_text: dict[int, dict[int, list[str]]] = {}
        for (p_idx, l_idx, is_space_before), char_pred in zip(char_locations, predictions, strict=True):
            if p_idx not in structured_text:
                structured_text[p_idx] = {}
            if l_idx not in structured_text[p_idx]:
                structured_text[p_idx][l_idx] = []

            if is_space_before:
                structured_text[p_idx][l_idx].append(" ")
            structured_text[p_idx][l_idx].append(str(char_pred))

        recognized_paragraphs: list[str] = []
        for p_idx in sorted(structured_text.keys()):
            para_lines = structured_text[p_idx]
            lines_str = ["".join(para_lines[l_idx]) for l_idx in sorted(para_lines.keys())]
            if lines_str:
                recognized_paragraphs.append("\n".join(lines_str))

        return "\n\n".join(recognized_paragraphs)
