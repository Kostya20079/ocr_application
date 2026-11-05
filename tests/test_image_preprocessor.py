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

    def test_find_projection_profile_segments(self):
        """Test 1D projection profile segmentation with gaps and noise thresholds."""
        profile = np.array([0, 0, 10, 10, 10, 0, 0, 0, 5, 5, 0], dtype=float)
        segments = ImagePreprocessor.find_projection_profile_segments(profile, threshold=0, min_size=2, min_gap=1)
        self.assertEqual(segments, [(2, 5), (8, 10)])

        # Test small gap merging
        profile_with_tiny_gap = np.array([10, 10, 0, 10, 10], dtype=float)
        merged = ImagePreprocessor.find_projection_profile_segments(profile_with_tiny_gap, threshold=0, min_size=2, min_gap=2)
        self.assertEqual(merged, [(0, 5)])


    def test_segment_paragraphs_on_sample(self):
        """Test paragraph segmentation on text_sample.png."""
        binary_img = ImagePreprocessor.otsu(self.image)
        fg_mask = ImagePreprocessor.invert_to_foreground_mask(binary_img)

        paragraphs = ImagePreprocessor.segment_paragraphs(fg_mask)
        self.assertIsInstance(paragraphs, list)
        self.assertGreater(len(paragraphs), 0)

        for para in paragraphs:
            self.assertIn("bbox", para)
            self.assertIn("crop", para)
            self.assertIn("line_spans", para)

            x, y, w, h = para["bbox"]
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)
            self.assertEqual(para["crop"].shape, (h, w))
            self.assertGreater(len(para["line_spans"]), 0)


if __name__ == "__main__":
    unittest.main()
