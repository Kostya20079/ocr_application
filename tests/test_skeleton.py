import unittest
from src.core.preprocessor import ImagePreprocessor
from src.engines import TemplateMatchingOCR, HogKnnOCR, TensorFlowOCR
from src.utils.helpers import get_resource_path
from src.utils.language import Language

class TestOCRSkeleton(unittest.TestCase):

    def test_preprocessor(self):
        class DummyImage:
            shape = (100, 100)
        gray = ImagePreprocessor.to_grayscale(DummyImage())
        self.assertEqual(gray.shape, (100, 100))

    def test_engines_instantiation(self):
        tm = TemplateMatchingOCR()
        hog = HogKnnOCR()
        tf = TensorFlowOCR()

        dummy_img = None
        self.assertIn("Template Matching", tm.recognize(dummy_img))
        self.assertIn("HOG + k-NN", hog.recognize(dummy_img))
        self.assertIn("TensorFlow CNN", tf.recognize(dummy_img))

    def test_resource_path(self):
        path = get_resource_path("data/samples")
        self.assertTrue(path.exists())

    def test_language_support(self):
        language = Language(default_locale="pl")
        self.assertEqual(language.locale, "pl")
        self.assertIn("Optycznego", language("app_title"))

        language.set_locale("en")
        self.assertEqual(language.locale, "en")
        self.assertIn("Optical Character", language("app_title"))

if __name__ == "__main__":
    unittest.main()
