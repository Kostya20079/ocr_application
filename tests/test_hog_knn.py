from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.engines.hog_knn import HogKnnOCR
from src.utils.helpers import get_resource_path


class TestHogKnnOCR(unittest.TestCase):
    """Unit and integration tests for HogKnnOCR engine."""

    @classmethod
    def setUpClass(cls):
        cls.engine = HogKnnOCR()
        cls.sample_path = get_resource_path("data/samples/text-sample2.png")
        if not cls.sample_path.exists():
            cls.sample_path = get_resource_path("data/samples/text_sample.png")
        if not cls.sample_path.exists():
            cls.sample_path = Path(__file__).resolve().parent.parent / "data" / "samples" / "text-sample2.png"

        if cls.sample_path.exists():
            cls.sample_image = cv2.imread(str(cls.sample_path))
        else:
            cls.sample_image = None

    def test_initialization_and_training_state(self):
        """Test engine initialization and model readiness."""
        self.assertEqual(self.engine.name, "HOG + k-NN")
        self.assertTrue(self.engine.is_trained)
        self.assertIsNotNone(self.engine.classifier)

    def test_extract_hog_features_shape(self):
        """Test HOG descriptor computation output shape and properties."""
        dummy_glyph = np.zeros((32, 32), dtype=np.uint8)
        dummy_glyph[8:24, 14:18] = 255  # simple vertical stroke

        features = self.engine.extract_hog_features(dummy_glyph)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.ndim, 1)
        self.assertEqual(len(features), 324)
        self.assertEqual(features.dtype, np.float32)

    def test_predict_glyph_returns_string(self):
        """Test predicting a single glyph crop."""
        dummy_glyph = np.zeros((32, 32), dtype=np.uint8)
        dummy_glyph[6:26, 14:18] = 255
        char = self.engine.predict_glyph(dummy_glyph)
        self.assertIsInstance(char, str)
        self.assertGreater(len(char), 0)

    def test_recognize_handles_none_and_invalid(self):
        """Test recognize graceful handling of None or bad input."""
        res_none = self.engine.recognize(None)
        self.assertIn("HOG + k-NN", res_none)

        res_invalid = self.engine.recognize("invalid-image-type")
        self.assertIn("Invalid image format", res_invalid)

    def test_recognize_pil_image_and_numpy(self):
        """Test recognize accepts both PIL images and numpy arrays."""
        if self.sample_image is None:
            self.skipTest("Sample image not found")

        # Test with numpy array
        text_np = self.engine.recognize(self.sample_image)
        self.assertIsInstance(text_np, str)
        self.assertGreater(len(text_np), 100)

        # Test with PIL image
        pil_img = Image.fromarray(cv2.cvtColor(self.sample_image, cv2.COLOR_BGR2RGB))
        text_pil = self.engine.recognize(pil_img)
        self.assertEqual(text_np, text_pil)

    def test_save_and_load_model(self):
        """Test serializing and loading classifier to/from file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_model_path = Path(tmp_dir) / "test_hog_knn.pkl"
            self.engine.save(temp_model_path)
            self.assertTrue(temp_model_path.exists())

            new_engine = HogKnnOCR(model_path=temp_model_path)
            self.assertTrue(new_engine.is_trained)
            self.assertIsNotNone(new_engine.classifier)

            dummy_glyph = np.zeros((32, 32), dtype=np.uint8)
            char1 = self.engine.predict_glyph(dummy_glyph)
            char2 = new_engine.predict_glyph(dummy_glyph)
            self.assertEqual(char1, char2)

    def test_recognize_text_sample2(self):
        """Test recognition of sample image with text lines and numbers."""
        sample2_path = get_resource_path("data/samples/text-sample1.png")
        if not sample2_path.exists():
            sample2_path = Path(__file__).resolve().parent.parent / "data" / "samples" / "text-sample1.png"

        if not sample2_path.exists():
            self.skipTest("text-sample1.png not found")

        img = cv2.imread(str(sample2_path))
        text = self.engine.recognize(img)
        self.assertIn("OCR", text.upper())
        self.assertIn("12345", text)

    def test_train_script_execution(self):
        """Test training pipeline function saves artifact and executes cleanly."""
        from scripts.train_hog_knn import train_hog_knn

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_model_path = Path(tmp_dir) / "test_trained_model.pkl"
            out = train_hog_knn(
                output_model=temp_model_path,
                n_neighbors=1,
                test_sample=None,
            )
            self.assertEqual(out, temp_model_path)
            self.assertTrue(temp_model_path.exists())
            self.assertGreater(temp_model_path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
