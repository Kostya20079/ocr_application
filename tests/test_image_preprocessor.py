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

        paragraphs = ImagePreprocessor.segment_paragraphs(binary_img)
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

    def test_segment_lines_preserves_paragraph_hierarchy(self):
        """Test segment_lines on 2D binary array preserves sequential ordering of paragraphs."""
        binary_img = ImagePreprocessor.otsu(self.image)
        fg_mask = ImagePreprocessor.invert_to_foreground_mask(binary_img)

        lines = ImagePreprocessor.segment_lines(fg_mask)
        self.assertIsInstance(lines, list)
        self.assertEqual(len(lines), 29)

        # Verify paragraph_idx is non-decreasing (sequential rows per paragraph)
        para_indices = [line["paragraph_idx"] for line in lines]
        self.assertEqual(para_indices, sorted(para_indices))
        # Ensure we detected multiple paragraphs
        self.assertGreater(len(set(para_indices)), 1)

    def test_segment_characters_and_spaces(self):
        """Test segment_characters_and_spaces on a segmented line crop."""
        binary_img = ImagePreprocessor.otsu(self.image)

        lines = ImagePreprocessor.segment_lines(binary_img)
        sample_line = lines[2]

        characters = ImagePreprocessor.segment_characters_and_spaces(sample_line["crop"])
        self.assertIsInstance(characters, list)
        self.assertGreater(len(characters), 0)

        for char in characters:
            self.assertIn("bbox", char)
            self.assertIn("crop", char)
            self.assertIn("is_space_before", char)

            x, y, w, h = char["bbox"]
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)
            self.assertEqual(char["crop"].shape, (h, w))

        spaces_count = sum(1 for c in characters if c["is_space_before"])
        self.assertGreater(spaces_count, 0)

    def test_normalize_character(self):
        """Test character normalization to target fixed canvas dimensions."""
        binary_img = ImagePreprocessor.otsu(self.image)
        lines = ImagePreprocessor.segment_lines(binary_img)
        characters = ImagePreprocessor.segment_characters_and_spaces(lines[2]["crop"])
        first_char = characters[0]

        normalized = ImagePreprocessor.normalize_character(first_char["crop"], target_size=(32, 32), padding=2)

        self.assertIsInstance(normalized, np.ndarray)
        self.assertEqual(normalized.shape, (32, 32))
        self.assertEqual(normalized.dtype, np.uint8)

        # Ensure output is strictly binary (0 and 255) and contains foreground text pixels
        unique_vals = set(np.unique(normalized))
        self.assertTrue(unique_vals.issubset({0, 255}))
        self.assertIn(255, unique_vals)
        self.assertIn(0, unique_vals)

        # Empty crop test
        empty_crop = np.zeros((10, 10), dtype=np.uint8)
        norm_empty = ImagePreprocessor.normalize_character(empty_crop, target_size=(32, 32))
        self.assertEqual(norm_empty.shape, (32, 32))
        self.assertEqual(np.count_nonzero(norm_empty), 0)


if __name__ == "__main__":
    unittest.main()
