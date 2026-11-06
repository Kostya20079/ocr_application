"""Visual test script to inspect paragraph segmentation results on data/samples/text_sample.png.

Draws bounding boxes and labels around each segmented paragraph and displays
the visual overlay alongside the binary foreground mask using matplotlib.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.core.preprocessor import ImagePreprocessor
from src.utils.helpers import get_resource_path

colors = [
    (255, 0, 0),  # Red
    (0, 180, 0),  # Green
    (0, 100, 255),  # Blue
    (255, 140, 0),  # Orange
    (148, 0, 211),  # Purple
    (0, 200, 200),  # Cyan
]

font_scale = 0.6
thickness = 1

def visualize_segmented_paragraphs(image):

    binary = ImagePreprocessor.otsu(image)

    paragraphs = ImagePreprocessor.segment_paragraphs(binary)
    print(f"Detected {len(paragraphs)} paragraph(s).")

    # Convert original to RGB for matplotlib display
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        vis_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for idx, para in enumerate(paragraphs):
        x, y, w, h = para["bbox"]
        color = colors[idx % len(colors)]
        num_lines = len(para["line_spans"])
        print(f"Paragraph {idx + 1}: BBox (x={x}, y={y}, w={w}, h={h}), Lines={num_lines}")

        # Draw bounding rectangle
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)

        # Label above paragraph box
        label = f"Para {idx + 1} ({num_lines} lines)"
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

    fg_mask = ImagePreprocessor.invert_to_foreground_mask(binary)

    axes[0].imshow(fg_mask, cmap="gray")
    axes[0].set_title("Normalized Foreground Mask")
    axes[0].axis("off")

    axes[1].imshow(vis_image)
    axes[1].set_title(f"Segmented Paragraphs ({len(paragraphs)} detected)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_segmented_lines(image):
    binary = ImagePreprocessor.otsu(image)

    lines = ImagePreprocessor.segment_lines(binary)
    print(f"Detected {len(lines)} line(s)")

    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        vis_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for idx, line in enumerate(lines):
        x, y, w, h = line["bbox"]
        p_idx = line.get("paragraph_idx", 0)
        color = colors[p_idx % len(colors)]

        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)

        label = f"P{p_idx}:L{idx + 1}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        label_y = max(y - 2, th + 2)
        label_x = max(x, 0)
        cv2.rectangle(
            vis_image,
            (label_x, label_y - th - 2),
            (label_x + tw + 4, label_y + baseline),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            vis_image,
            label,
            (label_x + 2, label_y - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    fg_mask = ImagePreprocessor.invert_to_foreground_mask(binary)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    axes[0].imshow(fg_mask, cmap="gray")
    axes[0].set_title("Normalized Foreground Mask")
    axes[0].axis("off")

    axes[1].imshow(vis_image)
    axes[1].set_title(f"Segmented Lines ({len(lines)} detected)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_segmented_characters(image, line_idx: int = 2):
    binary = ImagePreprocessor.otsu(image)
    fg_mask = ImagePreprocessor.invert_to_foreground_mask(binary)

    lines = ImagePreprocessor.segment_lines(fg_mask)

    target_line = lines[line_idx]
    line_crop = target_line["crop"]
    characters = ImagePreprocessor.segment_characters_and_spaces(line_crop)

    print(f"Visualizing Line {line_idx + 1}: Found {len(characters)} characters, "
          f"{sum(1 for c in characters if c['is_space_before'])} spaces")

    vis_line = cv2.cvtColor(line_crop, cv2.COLOR_GRAY2RGB)

    for c_idx, char in enumerate(characters):
        cx, cy, cw, ch = char["bbox"]
        cv2.rectangle(vis_line, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 1)

        if char["is_space_before"]:
            cv2.line(vis_line, (max(0, cx - 2), 0), (max(0, cx - 2), line_crop.shape[0]), (255, 140, 0), 2)

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.imshow(vis_line)
    ax.set_title(f"Segmented Characters & Spaces on Line {line_idx + 1} (Green=Glyph, Orange=Space)")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_normalized_word_letters(image, line_idx: int = 1, target_size: tuple[int, int] = (32, 32)):
    binary = ImagePreprocessor.otsu(image)
    fg_mask = ImagePreprocessor.invert_to_foreground_mask(binary)

    lines = ImagePreprocessor.segment_lines(fg_mask)
    if not lines or line_idx >= len(lines):
        print(f"Line index {line_idx} out of range (found {len(lines)} lines)")
        return

    target_line = lines[line_idx]
    characters = ImagePreprocessor.segment_characters_and_spaces(target_line["crop"])

    # Extract letters of the first word (all characters before the first space after index 0)
    word_chars = []
    for idx, char in enumerate(characters):
        if idx > 0 and char["is_space_before"]:
            break
        word_chars.append(char)

    if not word_chars:
        print("No characters found in the first word.")
        return

    num_chars = len(word_chars)
    fig, axes = plt.subplots(2, num_chars, figsize=(3 * num_chars, 6))
    if num_chars == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for i, char in enumerate(word_chars):
        raw_crop = char["crop"]
        norm_crop = ImagePreprocessor.normalize_character(raw_crop, target_size=target_size)

        axes[0, i].imshow(raw_crop, cmap="gray")
        axes[0, i].set_title(f"Raw '{i+1}'\n{raw_crop.shape[1]}x{raw_crop.shape[0]}")
        axes[0, i].axis("off")

        axes[1, i].imshow(norm_crop, cmap="gray")
        axes[1, i].set_title(f"Norm '{i+1}'\n{target_size[0]}x{target_size[1]}")
        axes[1, i].axis("off")

    plt.suptitle("First Word Glyphs: Raw Crop (Top) vs Normalized (Bottom)", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    img_name = "text_sample.png"
    sample_path = get_resource_path(f"data/samples/{img_name}")
    if not sample_path.exists():
        sample_path = get_resource_path("data/samples/text_sample1.png")

    image = cv2.imread(str(sample_path))
    if image is None:
        raise ValueError(f"Failed to read image at {sample_path}")

    visualize_segmented_paragraphs(image)
    visualize_segmented_lines(image)
    visualize_segmented_characters(image, 1)
    visualize_normalized_word_letters(image, 1)
