from typing import Any

import numpy as np


class ImagePreprocessor:
    """Preprocesses input images (grayscale conversion, binarization, denoising, segmentation)."""

    @staticmethod
    def to_grayscale(image: Any) -> Any:
        """Converts an image to grayscale."""
        if hasattr(image, "shape") and len(image.shape) == 2:
            return image
        try:
            import numpy as np
            if isinstance(image, np.ndarray):
                return np.mean(image[:, :, :3], axis=2).astype(np.uint8)
        except ImportError:
            pass
        return image

    @staticmethod
    def otsu(image: Any) -> Any:
        """Applies Otsu's automatic thresholding to binarize an image.

        Otsu's method finds the optimal threshold by iterating over all possible
        gray-level thresholds (0-255) and selecting the one that maximizes
        the between-class variance:
            sigma_b^2 = Wb * Wf * (mub - muf)^2
        where Wb and Wf are background/foreground probabilities, and mub, muf
        are their respective class mean intensities.

        Args:
            image: Input image (grayscale or RGB numpy array / PIL image).

        Returns:
            Binarized numpy array with foreground/background values (0 or 255).
        """
        gray = ImagePreprocessor.to_grayscale(image)

        # Calculate histogram
        his = np.bincount(gray.ravel(), minlength=256)

        pixel_number = gray.size
        mean_weight = 1.0 / pixel_number

        intensity_arr = np.arange(256)

        final_thresh = 0
        final_value = -1

        # Test every possible threshold
        for t in range(255):

            # Background
            pcb = np.sum(his[:t + 1])

            # Foreground
            pcf = np.sum(his[t + 1:])

            # Avoid division by zero
            if pcb == 0 or pcf == 0:
                continue

            # Class probabilities
            Wb = pcb * mean_weight
            Wf = pcf * mean_weight

            # Class means
            mub = np.sum(intensity_arr[:t + 1] * his[:t + 1]) / pcb

            muf = np.sum(intensity_arr[t + 1:] * his[t + 1:]) / pcf

            # Between-class variance
            value = Wb * Wf * (mub - muf) ** 2

            if value > final_value:
                final_thresh = t
                final_value = value

        # Apply threshold
        final_img = np.zeros_like(gray)
        final_img[gray > final_thresh] = 255

        return final_img

    @staticmethod
    def invert_to_foreground_mask(binary_image: Any) -> Any:
        """Ensures text/ink pixels are foreground (255) and background paper is 0.

        In scanned documents and book pages, text is typically dark (low values)
        on light paper (high values). After thresholding with `gray > threshold = 255`,
        the background paper becomes 255 and text becomes 0.
        
        To perform mathematical projection profiles, morphological operations,
        and template matching, foreground text pixels must be represented by 255
        and background paper by 0.

        Args:
            binary_image: Binary image (0 and 255).

        Returns:
            Normalized binary foreground mask where text=255 and background=0.
        """
        if not isinstance(binary_image, np.ndarray):
            binary_image = np.array(binary_image, dtype=np.uint8)

        white_pixels = np.count_nonzero(binary_image == 255)
        black_pixels = np.count_nonzero(binary_image == 0)

        # In standard documents, background covers the majority of the page (> 50%)
        # If white pixels dominate, the background is 255, so we invert
        if white_pixels > black_pixels:
            return np.where(binary_image == 255, 0, 255).astype(np.uint8)

        return binary_image.astype(np.uint8)

    @staticmethod
    def find_projection_profile_segments(
        profile: np.ndarray,
        threshold: float = 0.0,
        min_size: int = 2,
        min_gap: int = 1,
    ) -> list[tuple[int, int]]:
        """Segments a 1D projection profile into contiguous active intervals [start, end).

        Identifies regions where the profile exceeds `threshold`, merging small gaps
        narrower than `min_gap` and filtering out noise segments shorter than `min_size`.

        Args:
            profile: 1D numpy array representing projection profile sums.
            threshold: Value above which a position is considered active content.
            min_size: Minimum length of a valid segment (to discard single-pixel noise).
            min_gap: Minimum blank gap length required to separate two distinct segments.

        Returns:
            List of (start, end) coordinate tuples.
        """
        active = profile > threshold
        if not np.any(active):
            return []

        # Find transitions: rising edges (+1) and falling edges (-1)
        diff = np.diff(active.astype(np.int8))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        if active[0]:
            starts = np.insert(starts, 0, 0)
        if active[-1]:
            ends = np.append(ends, len(profile))

        raw_segments = list(zip(starts, ends))
        if not raw_segments:
            return []

        # Merge segments separated by tiny gaps smaller than min_gap
        merged_segments: list[tuple[int, int]] = []
        cur_start, cur_end = raw_segments[0]

        for next_start, next_end in raw_segments[1:]:
            gap = next_start - cur_end
            if gap < min_gap:
                cur_end = next_end
            else:
                if (cur_end - cur_start) >= min_size:
                    merged_segments.append((cur_start, cur_end))
                cur_start, cur_end = next_start, next_end

        if (cur_end - cur_start) >= min_size:
            merged_segments.append((cur_start, cur_end))

        return merged_segments

    @staticmethod
    def segment_paragraphs(
        foreground_mask: np.ndarray,
        gap_multiplier: float = 2.5,
        min_line_height: int = 5,
        indent_threshold_ratio: float = 0.035,
        short_line_threshold_ratio: float = 0.12,
    ) -> list[dict[str, Any]]:
        """Segments a document page into individual paragraphs using multi-feature layout analysis.

        In book and document typography, paragraphs are structured using:
        1. Vertical spacing: headings or section breaks have significantly larger line gaps.
        2. First-line indents (Alinea): new paragraphs frequently begin with an indented first line.
        3. Short line endings: the terminal line of a paragraph terminates early, leaving whitespace
           before the right margin.

        This method:
        1. Detects all text lines via horizontal projection profile.
        2. Calculates horizontal boundaries (x_left, x_right) for each line.
        3. Infers page column bounds (left_margin, right_margin, column_width).
        4. Detects paragraph boundaries whenever:
           - Vertical gap exceeds `median_gap * gap_multiplier` (vertical spacing break), OR
           - Next line is indented to the right by at least `indent_threshold_ratio * column_width`, OR
           - Current line ends prematurely by at least `short_line_threshold_ratio * column_width`.
        5. Returns tight bounding boxes and line spans per paragraph.

        Args:
            foreground_mask: Binary 2D array where text pixels are 255 and background is 0.
            gap_multiplier: Factor above median line spacing to identify a vertical paragraph break.
            min_line_height: Minimum pixel height to qualify as a valid text line.
            indent_threshold_ratio: Ratio of column width required to trigger an indent-based break.
            short_line_threshold_ratio: Ratio of column width required to consider a line ending short.

        Returns:
            List of dictionaries per paragraph:
                {
                    "bbox": (x, y, width, height),
                    "crop": 2D np.ndarray paragraph crop,
                    "line_spans": list of (line_y_start, line_y_end) relative to page
                }
        """
        # Horizontal projection profile: sum pixel intensities along rows
        h_profile = np.sum(foreground_mask > 0, axis=1)

        # Detect individual line candidates (lines separated by blank rows)
        raw_lines = ImagePreprocessor.find_projection_profile_segments(
            h_profile,
            threshold=0,
            min_size=min_line_height,
            min_gap=1,
        )

        if not raw_lines:
            return []

        # Analyze horizontal span for each detected line
        line_info: list[dict[str, Any]] = []
        for y_s, y_e in raw_lines:
            line_slice = foreground_mask[y_s:y_e, :]
            fg_cols = np.where(np.sum(line_slice > 0, axis=0) > 0)[0]
            if fg_cols.size == 0:
                continue
            x_s = int(fg_cols[0])
            x_e = int(fg_cols[-1]) + 1
            line_info.append({
                "y_span": (y_s, y_e),
                "x_span": (x_s, x_e),
                "height": y_e - y_s,
                "width": x_e - x_s,
            })

        if not line_info:
            return []

        if len(line_info) == 1:
            line_groups = [[line_info[0]["y_span"]]]
        else:
            # Estimate document column margins
            all_x_lefts = [l["x_span"][0] for l in line_info]
            all_x_rights = [l["x_span"][1] for l in line_info]

            # Left margin is estimated using low percentile (ignoring occasional outdents)
            column_left = float(np.percentile(all_x_lefts, 15))
            column_right = float(np.percentile(all_x_rights, 85))
            column_width = max(1.0, column_right - column_left)

            # Thresholds for horizontal layout cues
            indent_thresh = max(12.0, column_width * indent_threshold_ratio)
            short_line_thresh = max(20.0, column_width * short_line_threshold_ratio)

            # Compute median vertical gap between consecutive lines
            gaps = [
                line_info[i + 1]["y_span"][0] - line_info[i]["y_span"][1]
                for i in range(len(line_info) - 1)
            ]
            positive_gaps = [g for g in gaps if g > 0]
            median_gap = float(np.median(positive_gaps)) if positive_gaps else 1.0

            # Vertical spacing break threshold
            vertical_gap_thresh = max(median_gap * gap_multiplier, median_gap + 4)

            line_groups = []
            current_group = [line_info[0]["y_span"]]

            for i in range(len(line_info) - 1):
                cur_line = line_info[i]
                next_line = line_info[i + 1]

                v_gap = next_line["y_span"][0] - cur_line["y_span"][1]

                # Large vertical gap (heading / section break)
                is_vertical_break = v_gap > vertical_gap_thresh

                # Next line is indented to the right (first-line indent)
                is_indent_break = (next_line["x_span"][0] - column_left) > indent_thresh

                # Current line terminates early before column right edge
                is_short_line_break = (column_right - cur_line["x_span"][1]) > short_line_thresh

                # Trigger new paragraph if vertical gap, or if indent/short-line cue occurs with normal spacing
                if is_vertical_break or is_indent_break or is_short_line_break:
                    line_groups.append(current_group)
                    current_group = [next_line["y_span"]]
                else:
                    current_group.append(next_line["y_span"])

            line_groups.append(current_group)

        paragraphs: list[dict[str, Any]] = []

        for group in line_groups:
            y_start = group[0][0]
            y_end = group[-1][1]

            # Slice paragraph vertically
            para_slice = foreground_mask[y_start:y_end, :]

            # Compute vertical projection profile on the paragraph to find horizontal bounds
            v_profile = np.sum(para_slice > 0, axis=0)
            fg_cols = np.where(v_profile > 0)[0]

            if fg_cols.size == 0:
                continue

            x_start = int(fg_cols[0])
            x_end = int(fg_cols[-1]) + 1

            tight_crop = para_slice[:, x_start:x_end]

            paragraphs.append(
                {
                    "bbox": (x_start, y_start, x_end - x_start, y_end - y_start),
                    "crop": tight_crop,
                    "line_spans": group,
                }
            )

        return paragraphs
