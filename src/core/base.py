from abc import ABC, abstractmethod
from typing import Any

class BaseOCREngine(ABC):
    """Base interface for all OCR recognition engines."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def recognize(self, image: Any) -> str:
        """
        Process the input image and return the recognized text.

        :param image: Input image as numpy ndarray or PIL.Image.
        :return: Recognized text string.
        """
        pass
