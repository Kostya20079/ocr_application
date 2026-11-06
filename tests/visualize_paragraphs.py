"""Visual test script to inspect paragraph segmentation results on data/samples/text_sample.png.

Draws bounding boxes and labels around each segmented paragraph and displays
the visual overlay alongside the binary foreground mask using matplotlib.
"""

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.core.preprocessor import ImagePreprocessor
from src.utils.helpers import get_resource_path


def visualize_segmented_paragraphs():
    sample_path = get_resource_path("data/samples/text_sample1.png")
    if not sample_path.exists():
        sample_path = Path(__file__).resolve().parent.parent / "data" / "samples" / "text_sample.png"

    image = cv2.imread(str(sample_path))
    if image is None:
        raise ValueError(f"Failed to read image at {sample_path}")

    binary = ImagePreprocessor.otsu(image)
    fg_mask = ImagePreprocessor.invert_to_foreground_mask(binary)

    paragraphs = ImagePreprocessor.segment_paragraphs(fg_mask)
    print(f"Detected {len(paragraphs)} paragraph(s).")

    # Convert original to RGB for matplotlib display
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        vis_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Distinct colors for multiple paragraphs (RGB)
    colors = [
        (255, 0, 0), # Red
        (0, 180, 0), # Green
        (0, 100, 255), # Blue
        (255, 140, 0), # Orange
        (148, 0, 211), # Purple
        (0, 200, 200), # Cyan
    ]

    for idx, para in enumerate(paragraphs):
        x, y, w, h = para["bbox"]
        color = colors[idx % len(colors)]
        num_lines = len(para["line_spans"])
        print(f"Paragraph {idx + 1}: BBox (x={x}, y={y}, w={w}, h={h}), Lines={num_lines}")

        # Draw bounding rectangle
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)

        # Label above paragraph box
        label = f"Para {idx + 1} ({num_lines} lines)"
        font_scale = 0.6
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        label_y = max(y - 8, text_h + 4)
        # Background badge for text label for readability
        cv2.rectangle(
            vis_image,
            (x, label_y - text_h - 4),
            (x + text_w + 6, label_y + baseline),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            vis_image,
            label,
            (x + 3, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    axes[0].imshow(fg_mask, cmap="gray")
    axes[0].set_title("Normalized Foreground Mask (Step 3)")
    axes[0].axis("off")

    axes[1].imshow(vis_image)
    axes[1].set_title(f"Segmented Paragraphs ({len(paragraphs)} detected)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_segmented_paragraphs()
