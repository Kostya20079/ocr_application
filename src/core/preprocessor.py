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
