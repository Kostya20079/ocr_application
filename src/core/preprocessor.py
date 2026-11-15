from typing import Any

import cv2
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
        total = white_pixels + black_pixels

        white_ratio = white_pixels / max(1, total)

        if white_ratio > 0.5:
            return np.where(binary_image == 255, 0, 255).astype(np.uint8)

        return binary_image.astype(np.uint8)

    @staticmethod
    def find_projection_profile_segments(
        profile: np.ndarray,
        threshold: float = 0.0,
        min_size: int = 2,
        min_gap: int = 0.7,
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
        binary_image: np.ndarray,
        gap_multiplier: float = 2.5,
        min_line_height: int = 5,
        indent_threshold_ratio: float = 0.035,
        short_line_threshold_ratio: float = 0.12,
    ) -> list[dict[str, Any]]:
        """Segments a 2D binary document image into individual paragraphs using multi-feature layout analysis.

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
            binary_image: 2D binary numpy array (text pixels=255, background=0).
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
        foreground_mask = ImagePreprocessor.invert_to_foreground_mask(binary_image)

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

            median_line_height = float(np.median([l["height"] for l in line_info]))

            vertical_gap_thresh = max(
                median_gap * gap_multiplier,
                median_gap + 4,
                median_line_height * 0.5,
            )

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

    @staticmethod
    def segment_lines(
        binary_image: np.ndarray,
        min_line_height: int = 5,
        min_gap: int = 1,
    ) -> list[dict[str, Any]]:
        """Segments a 2D binary document image into ordered text lines across paragraphs.

        Runs `segment_paragraphs()` first to preserve the sequential reading order
        of lines within each paragraph.

        Args:
            binary_image: 2D binary numpy array (text pixels=255, background=0).
            min_line_height: Minimum pixel height for a line.
            min_gap: Minimum blank rows to separate lines.

        Returns:
            List of dictionaries for each line:
                {
                    "bbox": (x, y, w, h),  # Global coordinates in page
                    "crop": 2D np.ndarray line crop,
                    "y_span": (y_start, y_end),  # Local to paragraph
                    "paragraph_idx": int
                }
        """
        foreground_mask = ImagePreprocessor.invert_to_foreground_mask(binary_image)
        paragraphs = ImagePreprocessor.segment_paragraphs(foreground_mask)
        all_lines: list[dict[str, Any]] = []

        for p_idx, para in enumerate(paragraphs):
            para_crop = para["crop"]
            para_x, para_y, _, _ = para["bbox"]

            # Compute horizontal projection profile across the paragraph crop
            h_profile = np.sum(para_crop > 0, axis=1)
            raw_spans = ImagePreprocessor.find_projection_profile_segments(
                h_profile,
                threshold=0,
                min_size=min_line_height,
                min_gap=min_gap,
            )

            for y_s, y_e in raw_spans:
                line_slice = para_crop[y_s:y_e, :]
                v_profile = np.sum(line_slice > 0, axis=0)
                fg_cols = np.where(v_profile > 0)[0]
                if fg_cols.size == 0:
                    continue
                x_s = int(fg_cols[0])
                x_e = int(fg_cols[-1]) + 1

                line_crop = line_slice[:, x_s:x_e]
                global_bbox = (int(para_x + x_s), int(para_y + y_s), int(x_e - x_s), int(y_e - y_s))

                all_lines.append({
                    "bbox": global_bbox,
                    "crop": line_crop,
                    "y_span": (y_s, y_e),
                    "paragraph_idx": p_idx,
                })

        return all_lines

    @staticmethod
    def segment_characters_and_spaces(
        binary_line: np.ndarray,
        space_ratio_threshold: float = 0.5,
        min_char_width: int = 1,
        min_char_height: int = 2,
        min_aspect_ratio_for_thin: float = 2.0,
    ) -> list[dict[str, Any]]:
        """Segments a 2D binary text line into individual character glyphs and detects word spaces.

        Args:
            binary_line: 2D binary numpy array representing a single line crop (text=255, bg=0).
            space_ratio_threshold: Multiplier of median character width to trigger word space.
            min_char_width: Minimum width in pixels to filter out noise.
            min_char_height: Minimum height in pixels to filter out noise.
            min_aspect_ratio_for_thin: Minimum width/height ratio to keep glyphs
                shorter than min_char_height (preserves hyphens, underscores, etc.).

        Returns:
            List of character dictionaries:
                {
                    "bbox": (x, y, w, h),      # Local to binary_line crop
                    "crop": 2D np.ndarray,     # Tight crop of the character glyph
                    "is_space_before": bool,   # True if a word space precedes this character
                }
        """
        line_mask = ImagePreprocessor.invert_to_foreground_mask(binary_line)
        if not np.any(line_mask > 0):
            return []

        line_h, line_w = line_mask.shape[:2]
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(line_mask, connectivity=8)
        if num_labels <= 1:
            return []

        raw_comps: list[dict[str, Any]] = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < 2 or (w < min_char_width and h < min_char_height):
                continue
            comp_mask = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
            raw_comps.append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "area": int(area),
                "mask": comp_mask,
            })

        if not raw_comps:
            return []

        # Compute median character width and height dynamically based on line height
        normal_comps = [
            c for c in raw_comps
            if c["h"] >= max(5, int(line_h * 0.25)) and 0.2 <= (c["w"] / max(1, c["h"])) <= 1.6
        ]
        if normal_comps:
            med_w = float(np.median([c["w"] for c in normal_comps]))
            med_h = float(np.median([c["h"] for c in normal_comps]))
        else:
            med_w = float(np.median([c["w"] for c in raw_comps])) if raw_comps else 12.0
            med_h = float(np.median([c["h"] for c in raw_comps])) if raw_comps else 14.0

        # Split touching glyphs (e.g. 'Th' or serifs touching)
        split_comps: list[dict[str, Any]] = []

        def split_component(c: dict[str, Any]) -> list[dict[str, Any]]:
            is_wide_cap = (c["h"] >= med_h * 1.25 and c["w"] >= med_w * 1.35)
            is_very_wide = (c["w"] >= med_w * 2.2 and c["w"] >= 20)
            if not (is_wide_cap or is_very_wide):
                return [c]

            v_prof = np.sum(c["mask"] > 0, axis=0)

            # Check if this glyph is 'm' (three prominent vertical peaks and height ~ med_h)
            if c["h"] <= int(med_h * 1.18) and c["w"] <= int(med_w * 2.1):
                # Count peaks in v_prof that reach near character height
                peak_thresh = int(c["h"] * 0.6)
                peaks = []
                for p_i in range(1, len(v_prof) - 1):
                    if v_prof[p_i] >= peak_thresh and v_prof[p_i] >= v_prof[p_i - 1] and v_prof[p_i] >= v_prof[p_i + 1]:
                        if not peaks or (p_i - peaks[-1]) >= 4:
                            peaks.append(p_i)
                if len(peaks) >= 3:
                    return [c]  # It is 'm', keep intact

            s = int(c["w"] * 0.25)
            e = int(c["w"] * 0.75)
            if e <= s:
                return [c]

            sub_prof = v_prof[s:e]
            min_val = int(np.min(sub_prof))
            max_valley_thresh = max(3, int(med_h * 0.25))
            if min_val > max_valley_thresh and not (c["w"] >= med_w * 2.5):
                return [c]

            min_indices = np.where(sub_prof == min_val)[0] + s
            valley_idx = int((min_indices[0] + min_indices[-1]) // 2) + 1
            min_char_extent = max(5, int(med_w * 0.35))
            if valley_idx < min_char_extent or (c["w"] - valley_idx) < min_char_extent:
                return [c]

            m1 = c["mask"][:, :valley_idx]
            m2 = c["mask"][:, valley_idx:]
            sub_results: list[dict[str, Any]] = []

            for m, off in [(m1, 0), (m2, valley_idx)]:
                fg = np.argwhere(m > 0)
                if fg.size == 0:
                    continue
                y1, x1 = fg.min(axis=0)
                y2, x2 = fg.max(axis=0) + 1
                sub_c = {
                    "x": c["x"] + off + int(x1),
                    "y": c["y"] + int(y1),
                    "w": int(x2 - x1),
                    "h": int(y2 - y1),
                    "area": len(fg),
                    "mask": m[y1:y2, x1:x2],
                }
                # Recursively split if still exceptionally wide (e.g. 3+ letters joined)
                if sub_c["w"] >= max(26, int(med_w * 2.2)):
                    sub_results.extend(split_component(sub_c))
                else:
                    sub_results.append(sub_c)

            return sub_results if sub_results else [c]

        for c in raw_comps:
            split_comps.extend(split_component(c))

        # Sort candidate components from left to right
        split_comps.sort(key=lambda comp: (comp["x"], comp["y"]))

        # Group vertically aligned multi-component glyphs ('i', 'j', ':', ';', '!', '?', quotes)
        merged_glyphs: list[dict[str, Any]] = []
        used = set()

        for i, c1 in enumerate(split_comps):
            if i in used:
                continue
            found_partner = None

            for j, c2 in enumerate(split_comps):
                if i == j or j in used:
                    continue

                c1_mid = c1["x"] + c1["w"] / 2.0
                c2_mid = c2["x"] + c2["w"] / 2.0
                horiz_dist = abs(c1_mid - c2_mid)
                overlap = min(c1["x"] + c1["w"], c2["x"] + c2["w"]) - max(c1["x"], c2["x"])

                # 'i' or 'j': dot above stem
                is_dot_above = (
                    c2["h"] <= max(7, int(line_h * 0.2)) and c2["w"] <= max(8, int(med_w * 0.6))
                    and c2["y"] < c1["y"]
                    and (c1["y"] - (c2["y"] + c2["h"])) <= max(10, int(line_h * 0.25))
                    and (horiz_dist <= max(6, int(med_w * 0.4)) or overlap > 0)
                )
                is_dot_below = (
                    c1["h"] <= max(7, int(line_h * 0.2)) and c1["w"] <= max(8, int(med_w * 0.6))
                    and c1["y"] < c2["y"]
                    and (c2["y"] - (c1["y"] + c1["h"])) <= max(10, int(line_h * 0.25))
                    and (horiz_dist <= max(6, int(med_w * 0.4)) or overlap > 0)
                )
                # ':' colon or ';' semicolon
                is_colon = (
                    c1["w"] <= max(7, int(med_w * 0.6)) and c2["w"] <= max(7, int(med_w * 0.6))
                    and c1["h"] <= max(8, int(line_h * 0.22)) and c2["h"] <= max(10, int(line_h * 0.25))
                    and abs(c1["x"] - c2["x"]) <= max(5, int(med_w * 0.4))
                    and int(line_h * 0.15) <= abs(c1["y"] - c2["y"]) <= int(line_h * 0.55)
                )
                # Quotation marks (twin strokes high in line)
                is_twin_quotes = (
                    c1["y"] <= int(line_h * 0.35) and c2["y"] <= int(line_h * 0.35)
                    and c1["w"] <= int(med_w * 0.8) and c2["w"] <= int(med_w * 0.8)
                    and abs(c1["x"] - c2["x"]) <= int(med_w * 1.2)
                    and abs(c1["y"] - c2["y"]) <= max(6, int(line_h * 0.15))
                )

                if is_dot_above or is_dot_below or is_colon or is_twin_quotes:
                    found_partner = j
                    break

            if found_partner is not None:
                c2 = split_comps[found_partner]
                used.add(i)
                used.add(found_partner)
                min_x = min(c1["x"], c2["x"])
                min_y = min(c1["y"], c2["y"])
                max_x = max(c1["x"] + c1["w"], c2["x"] + c2["w"])
                max_y = max(c1["y"] + c1["h"], c2["y"] + c2["h"])
                combined_crop = line_mask[min_y:max_y, min_x:max_x]
                merged_glyphs.append({
                    "bbox": (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y)),
                    "crop": combined_crop,
                    "x": min_x,
                })
            else:
                used.add(i)
                merged_glyphs.append({
                    "bbox": (int(c1["x"]), int(c1["y"]), int(c1["w"]), int(c1["h"])),
                    "crop": c1["mask"],
                    "x": c1["x"],
                })

        # Sort final characters strictly in reading order
        merged_glyphs.sort(key=lambda g: g["x"])

        # Detect whitespace gaps
        char_widths = [g["bbox"][2] for g in merged_glyphs]
        median_char_width = float(np.median(char_widths)) if char_widths else med_w

        inter_gaps = [
            merged_glyphs[k + 1]["bbox"][0] - (merged_glyphs[k]["bbox"][0] + merged_glyphs[k]["bbox"][2])
            for k in range(len(merged_glyphs) - 1)
        ]
        pos_gaps = [g for g in inter_gaps if g > 0]
        median_gap = float(np.median(pos_gaps)) if pos_gaps else 1.0

        space_threshold = max(
            median_char_width * space_ratio_threshold,
            median_gap * 2.2,
            5.0,
        )

        characters: list[dict[str, Any]] = []
        for idx, g in enumerate(merged_glyphs):
            if idx == 0:
                is_space_before = False
            else:
                prev_g = merged_glyphs[idx - 1]
                gap = g["bbox"][0] - (prev_g["bbox"][0] + prev_g["bbox"][2])
                is_space_before = gap >= space_threshold

            characters.append({
                "bbox": g["bbox"],
                "crop": g["crop"],
                "is_space_before": is_space_before,
            })

        return characters

    @staticmethod
    def normalize_character(
        char_crop: np.ndarray,
        target_size: tuple[int, int] = (32, 32),
        padding: int = 2,
        is_foreground: bool = True,
    ) -> np.ndarray:
        """Tightly crops foreground pixels and normalizes them onto a target canvas preserving aspect ratio.

        Ensures normalized character crops match the exact dimensions, foreground polarity (255=glyph, 0=bg),
        and aspect-ratio padding of the template dataset.

        Args:
            char_crop: 2D binary numpy array of character glyph.
            target_size: (width, height) of normalized output canvas.
            padding: Safe margin around the glyph inside target canvas.
            is_foreground: Whether char_crop is already a foreground mask (255=text, 0=bg). Defaults to True.

        Returns:
            Normalized 2D uint8 numpy array with shape (target_size[1], target_size[0]).
        """
        if is_foreground:
            crop_mask = np.asarray(char_crop, dtype=np.uint8)
        else:
            crop_mask = ImagePreprocessor.invert_to_foreground_mask(char_crop)
        target_w, target_h = target_size

        fg_indices = np.argwhere(crop_mask > 0)
        if fg_indices.size == 0:
            return np.zeros((target_h, target_w), dtype=np.uint8)

        y_min, x_min = fg_indices.min(axis=0)
        y_max, x_max = fg_indices.max(axis=0) + 1

        tight = crop_mask[y_min:y_max, x_min:x_max]
        crop_h, crop_w = tight.shape

        max_w = max(1, target_w - 2 * padding)
        max_h = max(1, target_h - 2 * padding)

        scale = min(max_w / max(crop_w, 1), max_h / max(crop_h, 1))
        new_w = max(1, int(round(crop_w * scale)))
        new_h = max(1, int(round(crop_h * scale)))

        resized = cv2.resize(tight, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        resized_binary = np.where(resized > 100, 255, 0).astype(np.uint8)

        canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        start_x = (target_w - new_w) // 2
        start_y = (target_h - new_h) // 2
        canvas[start_y: start_y + new_h, start_x: start_x + new_w] = resized_binary

        return canvas