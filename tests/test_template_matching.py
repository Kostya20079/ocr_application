
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.engines.template_matching import TemplateMatchingOCR


class TestTemplateMatchingOCR(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.templates_dir = Path(self.temp_dir.name)

        self.engine = TemplateMatchingOCR(
            templates_dir=self.templates_dir
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_template(self, character="I"):
        glyph = np.zeros((30, 8), dtype=np.uint8)
        glyph[:, :] = 255

        template = self.engine._normalize_character(glyph)

        template_path = self.templates_dir / "template_I.png"
        Image.fromarray(template).save(template_path)

        metadata = [
            {
                "char": character,
                "filename": template_path.name,
                "ascii_code": ord(character),
                "font_name": "test",
                "font_path": "",
                "width": 8,
                "height": 30,
            }
        ]

        index_path = self.templates_dir / "index.json"

        with index_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file)

        self.engine = TemplateMatchingOCR(
            templates_dir=self.templates_dir
        )

    def _create_text_image(self, positions):
        image = np.full((50, 80), 255, dtype=np.uint8)

        for x in positions:
            image[10:40, x:x + 8] = 0

        return image

    def test_engine_initialization(self):
        self.assertEqual(
            self.engine.name,
            "Template Matching"
        )

        self.assertEqual(
            self.engine.templates_dir,
            self.templates_dir
        )

    def test_missing_templates_directory(self):
        self.assertEqual(
            self.engine.templates,
            []
        )

    def test_missing_image(self):
        result = self.engine.recognize(None)

        self.assertIn("Template Matching", result)
        self.assertIn("No image provided", result)

    def test_missing_templates(self):
        image = np.full((50, 80), 255, dtype=np.uint8)

        result = self.engine.recognize(image)

        self.assertIn("Templates not found", result)

    def test_load_templates(self):
        self._create_template()

        self.assertEqual(len(self.engine.templates), 1)
        self.assertEqual(self.engine.templates[0][0], "I")

        self.assertEqual(
            self.engine.templates[0][1].shape,
            (32, 32)
        )

    def test_invalid_metadata(self):
        index_path = self.templates_dir / "index.json"
        index_path.write_text(
            "{invalid json",
            encoding="utf-8"
        )

        engine = TemplateMatchingOCR(
            templates_dir=self.templates_dir
        )

        self.assertEqual(engine.templates, [])

    def test_empty_metadata(self):
        index_path = self.templates_dir / "index.json"

        with index_path.open("w", encoding="utf-8") as file:
            json.dump([], file)

        engine = TemplateMatchingOCR(
            templates_dir=self.templates_dir
        )

        self.assertEqual(engine.templates, [])

    def test_preprocess_grayscale_image(self):
        image = np.full((50, 80), 255, dtype=np.uint8)
        image[10:40, 10:18] = 0

        result = self.engine._preprocess_image(image)

        self.assertIsNotNone(result)
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result[20, 12], 255)
        self.assertEqual(result[0, 0], 0)

    def test_preprocess_rgb_image(self):
        image = np.full((50, 80, 3), 255, dtype=np.uint8)
        image[10:40, 10:18] = 0

        result = self.engine._preprocess_image(image)

        self.assertIsNotNone(result)
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape, (50, 80))

    def test_preprocess_pil_image(self):
        image = Image.fromarray(
            np.full((50, 80), 255, dtype=np.uint8)
        )

        result = self.engine._preprocess_image(image)

        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (50, 80))

    def test_preprocess_invalid_input(self):
        self.assertIsNone(
            self.engine._preprocess_image("invalid image")
        )

    def test_preprocess_empty_image(self):
        image = np.array([], dtype=np.uint8)

        self.assertIsNone(
            self.engine._preprocess_image(image)
        )

    def test_find_segments(self):
        projection = np.array(
            [0, 0, 1, 1, 0, 1, 1, 1, 0]
        )

        result = self.engine._find_segments(projection)

        self.assertEqual(
            result,
            [(2, 4), (5, 8)]
        )

    def test_segment_lines(self):
        image = np.zeros((60, 50), dtype=np.uint8)

        image[5:15, 10:20] = 255
        image[35:45, 10:20] = 255

        lines = self.engine._segment_lines(image)

        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].shape[0], 10)
        self.assertEqual(lines[1].shape[0], 10)

    def test_segment_characters(self):
        line = np.zeros((30, 60), dtype=np.uint8)

        line[:, 5:13] = 255
        line[:, 30:38] = 255

        characters = self.engine._segment_characters(line)

        self.assertEqual(len(characters), 2)
        self.assertEqual(characters[0][1:], (5, 13))
        self.assertEqual(characters[1][1:], (30, 38))

    def test_normalize_character(self):
        character = np.full((30, 8), 255, dtype=np.uint8)

        result = self.engine._normalize_character(character)

        self.assertEqual(result.shape, (32, 32))
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.any(result == 255))
        self.assertTrue(np.any(result == 0))

    def test_normalize_empty_character(self):
        character = np.zeros((20, 20), dtype=np.uint8)

        result = self.engine._normalize_character(character)

        self.assertEqual(result.shape, (32, 32))
        self.assertFalse(np.any(result))

    def test_similarity_identical_templates(self):
        character = np.zeros((32, 32), dtype=np.uint8)
        character[5:27, 12:20] = 255

        score = self.engine._calculate_similarity(
            character,
            character.copy()
        )

        self.assertAlmostEqual(score, 1.0, places=5)

    def test_similarity_different_templates(self):
        first = np.zeros((32, 32), dtype=np.uint8)
        second = np.zeros((32, 32), dtype=np.uint8)

        first[5:27, 5:10] = 255
        second[5:27, 22:27] = 255

        score = self.engine._calculate_similarity(
            first,
            second
        )

        self.assertLess(score, 1.0)

    def test_match_template(self):
        self._create_template()

        character = np.zeros((30, 8), dtype=np.uint8)
        character[:, :] = 255

        normalized = self.engine._normalize_character(character)

        result = self.engine._match_template(normalized)

        self.assertEqual(result, "I")

    def test_recognize_single_character(self):
        self._create_template()

        image = self._create_text_image([10])

        result = self.engine.recognize(image)

        self.assertEqual(result, "I")

    def test_recognize_two_words(self):
        self._create_template()

        image = self._create_text_image([5, 40])

        result = self.engine.recognize(image)

        self.assertEqual(result, "I I")

    def test_recognize_multiple_lines(self):
        self._create_template()

        image = np.full((100, 80), 255, dtype=np.uint8)

        image[10:40, 10:18] = 0
        image[60:90, 10:18] = 0

        result = self.engine.recognize(image)

        self.assertEqual(result, "I\nI")

    def test_recognize_blank_image(self):
        self._create_template()

        image = np.full((50, 80), 255, dtype=np.uint8)

        result = self.engine.recognize(image)

        self.assertIn("No text detected", result)

    def test_recognize_invalid_input(self):
        self._create_template()

        result = self.engine.recognize("invalid image")

        self.assertIn("No text detected", result)

    def test_match_template_below_threshold(self):
        self._create_template(character="I")

        # Create a character pattern completely different from a vertical bar "I" (e.g., small horizontal bar)
        char = np.zeros((32, 32), dtype=np.uint8)
        char[0:2, :] = 255
        self.engine.MATCH_THRESHOLD = 0.95

        result = self.engine._match_template(char)
        self.assertEqual(result, "?")

    def test_default_templates_loading(self):
        default_engine = TemplateMatchingOCR()
        self.assertGreater(len(default_engine.templates), 0)
        self.assertEqual(default_engine.name, "Template Matching")

    def test_recognize_from_pil_image(self):
        self._create_template()
        image_arr = self._create_text_image([10])
        pil_img = Image.fromarray(image_arr)

        result = self.engine.recognize(pil_img)
        self.assertEqual(result, "I")

    def test_recognize_from_file_path(self):
        self._create_template()
        image_arr = self._create_text_image([10])
        img_path = self.templates_dir / "sample_test.png"
        Image.fromarray(image_arr).save(img_path)

        result = self.engine.recognize(img_path)
        self.assertEqual(result, "I")

    def test_recognize_multiple_distinct_templates(self):
        # Create template I (vertical bar) and template O (square box)
        glyph_i = np.zeros((30, 8), dtype=np.uint8)
        glyph_i[:, :] = 255
        tpl_i = self.engine._normalize_character(glyph_i)
        Image.fromarray(tpl_i).save(self.templates_dir / "template_I.png")

        glyph_o = np.zeros((30, 20), dtype=np.uint8)
        glyph_o[0:5, :] = 255
        glyph_o[-5:, :] = 255
        glyph_o[:, 0:5] = 255
        glyph_o[:, -5:] = 255
        tpl_o = self.engine._normalize_character(glyph_o)
        Image.fromarray(tpl_o).save(self.templates_dir / "template_O.png")

        metadata = [
            {"char": "I", "filename": "template_I.png"},
            {"char": "O", "filename": "template_O.png"},
        ]
        index_path = self.templates_dir / "index.json"
        with index_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f)

        engine = TemplateMatchingOCR(templates_dir=self.templates_dir)
        self.assertEqual(len(engine.templates), 2)

        # Image with I and O
        image = np.full((50, 100), 255, dtype=np.uint8)
        image[10:40, 10:18] = 0  # glyph I
        image[10:40, 30:50] = 255 - glyph_o  # glyph O

        result = engine.recognize(image)
        self.assertIn("I", result)
        self.assertIn("O", result)


if __name__ == "__main__":
    unittest.main()
