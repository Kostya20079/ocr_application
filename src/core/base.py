from abc import ABC, abstractmethod
from typing import Any


class BaseOCREngine(ABC):
    """Base interface for all OCR recognition engines."""

    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def is_ready(self) -> bool:
        """Returns True if the engine has loaded its required weights/templates and is ready to infer."""
        return True

    @abstractmethod
    def recognize(self, image: Any) -> str:
        """Process the input image and return the recognized text.

        Args:
            image: Input image as numpy ndarray or PIL.Image.

        Returns:
            Recognized text string.
        """
        pass

    def predict_glyph(self, glyph_crop: Any) -> str:
        """Predict character for a single glyph crop.

        Args:
            glyph_crop: 2D binary numpy array representing character glyph.

        Returns:
            Predicted character string.
        """
        raise NotImplementedError(f"{self.name} does not implement predict_glyph.")
