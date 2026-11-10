import sys
from pathlib import Path
from src.app.gui import run

if __name__ == '__main__':
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        image_path = Path(sys.argv[1])
        if image_path.is_file():
            from PIL import Image
            import numpy as np
            from src.engines import TemplateMatchingOCR

            engine = TemplateMatchingOCR()
            img = np.array(Image.open(image_path))
            result = engine.recognize(img)
            print("=" * 60)
            print(f"[OCR Console Output - {image_path.name}]")
            print("=" * 60)
            print(result)
            print("=" * 60)
            sys.exit(0)
    run()