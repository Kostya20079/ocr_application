from pathlib import Path
from typing import Any
from src.core.base import BaseOCREngine

class TemplateMatchingOCR(BaseOCREngine):
    """OCR engine based on Template Matching."""

    def __init__(self, templates_dir: Path | None = None) -> None:
        super().__init__(name="Template Matching")
        self.templates_dir = templates_dir

    def recognize(self, image: Any) -> str:
        # TODO: Implement template matching recognition
        return "[Template Matching: Recognition output in development]"
