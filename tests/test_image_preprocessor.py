import unittest
from pathlib import Path
import cv2
import numpy as np

from src.core.preprocessor import ImagePreprocessor
from src.utils.helpers import get_resource_path


class TestImagePreprocessor(unittest.TestCase):
    """Unit tests for Otsu's thresholding and grayscale conversion in ImagePreprocessor."""

    @classmethod
    def setUpClass(cls):
        cls.sample_path = get_resource_path("data/samples/text_sample.png")
        if not cls.sample_path.exists():
            cls.sample_path = Path(__file__).resolve().parent.parent / "data" / "samples" / "text_sample.png"

        if not cls.sample_path.exists():
            raise FileNotFoundError(f"Sample image not found at {cls.sample_path}")

        cls.image = cv2.imread(str(cls.sample_path))
        if cls.image is None:
            raise ValueError(f"Failed to load image from {cls.sample_path}")

    def test_to_grayscale_shape_and_type(self):
        """Test converting text_sample.png to grayscale."""
        gray = ImagePreprocessor.to_grayscale(self.image)
        self.assertIsNotNone(gray)
        self.assertEqual(len(gray.shape), 2)
        self.assertEqual(gray.shape[:2], self.image.shape[:2])
        self.assertEqual(gray.dtype, np.uint8)

    def test_otsu_output_properties(self):
        """Test Otsu binarization output dimensions, type, and binary values on text_sample.png."""
        binary_img = ImagePreprocessor.otsu(self.image)

        self.assertIsNotNone(binary_img)
        self.assertEqual(binary_img.shape, self.image.shape[:2])
        self.assertEqual(binary_img.dtype, np.uint8)

        # Output should only contain binary values (0 and 255)
        unique_values = set(np.unique(binary_img))
        self.assertTrue(unique_values.issubset({0, 255}))

        # Text sample should contain both foreground and background pixels
        self.assertIn(0, unique_values)
        self.assertIn(255, unique_values)

    def test_invert_to_foreground_mask(self):
        """Test foreground mask normalization where text is 255 and background is 0."""
        binary_img = ImagePreprocessor.otsu(self.image)
        fg_mask = ImagePreprocessor.invert_to_foreground_mask(binary_img)

        self.assertEqual(fg_mask.shape, binary_img.shape)
        self.assertEqual(fg_mask.dtype, np.uint8)

        # In a typical document page, background (0) dominates foreground text (255)
        num_bg = np.count_nonzero(fg_mask == 0)
        num_fg = np.count_nonzero(fg_mask == 255)
        self.assertGreater(num_bg, num_fg)


if __name__ == "__main__":
    unittest.main()
