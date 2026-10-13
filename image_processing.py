"""
Image Processing Module
NMIMS B.Tech - Image and Video Processing (IVP) Mini-Project
"Image Enhancement and Filtering Tool Using Python and OpenCV"

This module contains classical digital image processing algorithms:
1. Grayscale Conversion
2. Digital Image Negative
3. Contrast Stretching (Min-Max Linear Mapping)
4. Histogram Equalization (Grayscale & Luminance-based Color)
5. Image Smoothing (Gaussian Filter, Median Filter)
6. Edge Detection (Sobel Gradient, Canny Edge Detection)
7. Image Thresholding (Global Binary, Otsu's Thresholding)
8. Histogram Computation Utilities

All functions use OpenCV and NumPy without any deep learning or machine learning APIs.
"""

from typing import Tuple, Union, Dict, Any
import cv2
import numpy as np


def validate_image(image: np.ndarray) -> None:
    """Validate that the input is a valid non-empty NumPy image array."""
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid image: input must be a non-empty NumPy array.")
    if image.size == 0:
        raise ValueError("Invalid image: image array is empty.")


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR or RGB color image to grayscale.

    Formula (Standard NTSC/Rec.601 luminance equation):
        Y = 0.299*R + 0.587*G + 0.114*B

    Args:
        image (np.ndarray): Input image (BGR color or already grayscale).

    Returns:
        np.ndarray: Grayscale single-channel image (uint8).
    """
    validate_image(image)
    if len(image.shape) == 2:
        # Already grayscale
        return image.copy()
    elif len(image.shape) == 3:
        if image.shape[2] == 1:
            return image[:, :, 0].copy()
        # Convert BGR color image to single-channel Grayscale
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unsupported image dimensions for grayscale conversion.")


def image_negative(image: np.ndarray) -> np.ndarray:
    """
    Compute digital negative transformation of an image.

    Transformation function:
        s = (L - 1) - r
        Where L = 256 for 8-bit images, so s = 255 - r.

    Reverses intensity levels: dark areas become bright, bright areas become dark.

    Args:
        image (np.ndarray): Input image (grayscale or color).

    Returns:
        np.ndarray: Negative image of the same shape and type.
    """
    validate_image(image)
    # Digital negative transformation: subtract each pixel intensity from 255
    return 255 - image


def contrast_stretching(
    image: np.ndarray,
    r_min: int = None,
    r_max: int = None
) -> np.ndarray:
    """
    Apply linear contrast stretching (normalization) to expand the dynamic range
    of pixel intensities to span the full [0, 255] display range.

    Piecewise linear transformation formula:
        s = ((r - r_min) / (r_max - r_min)) * 255

    Pixels below r_min are clipped to 0, and pixels above r_max are clipped to 255.

    Args:
        image (np.ndarray): Input image (grayscale or color).
        r_min (int, optional): Lower intensity threshold. If None, uses image min.
        r_max (int, optional): Upper intensity threshold. If None, uses image max.

    Returns:
        np.ndarray: Contrast-stretched uint8 image.
    """
    validate_image(image)

    # Convert to float for accurate mathematical computation
    img_float = image.astype(np.float32)

    # Determine default min and max if not specified by user
    actual_min = float(np.min(image))
    actual_max = float(np.max(image))

    low = float(r_min) if r_min is not None else actual_min
    high = float(r_max) if r_max is not None else actual_max

    # Guard against division by zero (flat intensity images)
    if high <= low:
        return np.clip(image, 0, 255).astype(np.uint8)

    # Linear stretching formula with clipping
    stretched = ((img_float - low) / (high - low)) * 255.0
    stretched = np.clip(stretched, 0, 255)

    return stretched.astype(np.uint8)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Enhance global contrast using Histogram Equalization.

    Theoretical basis:
        Transforms the intensity probability density function (PDF) so that
        the resulting cumulative distribution function (CDF) is approximately linear.

    For grayscale images:
        Direct application of cv2.equalizeHist().

    For color images:
        Converts BGR to YCrCb color space, equalizes only the luminance channel (Y)
        to prevent color hue distortion, and converts back to BGR.

    Args:
        image (np.ndarray): Input image (grayscale or BGR color).

    Returns:
        np.ndarray: Histogram equalized image.
    """
    validate_image(image)

    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        gray = image if len(image.shape) == 2 else image[:, :, 0]
        return cv2.equalizeHist(gray)

    # Color image handling:
    # Equalizing R, G, B independently causes severe chromatic shift.
    # Textbook standard approach: convert to YCrCb (or LAB), equalize Y, recombine.
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)

    # Equalize luminance component
    y_equalized = cv2.equalizeHist(y_channel)

    # Merge channels back and convert to BGR
    merged_ycrcb = cv2.merge([y_equalized, cr_channel, cb_channel])
    return cv2.cvtColor(merged_ycrcb, cv2.COLOR_YCrCb2BGR)


def gaussian_smoothing(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Apply Gaussian Smoothing filter (Low-pass linear filter).

    Theoretical basis:
        Convolves the image with a 2D Gaussian kernel:
            G(x, y) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2))
        Weights nearby pixels more heavily than distant pixels, effectively
        reducing Gaussian noise while preserving edges better than a box blur.

    Args:
        image (np.ndarray): Input image.
        kernel_size (int): Size of the square kernel. Must be positive odd integer.
        sigma (float): Gaussian standard deviation in both X and Y directions.

    Returns:
        np.ndarray: Smoothed image.
    """
    validate_image(image)

    # Kernel size must be positive and odd
    if kernel_size <= 0:
        raise ValueError("Kernel size must be a positive integer.")
    if kernel_size % 2 == 0:
        kernel_size += 1  # Automatically make odd if even

    if sigma <= 0:
        sigma = 0.1  # OpenCV requires positive sigma or 0 for auto-calculation

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)


def median_smoothing(
    image: np.ndarray,
    kernel_size: int = 5
) -> np.ndarray:
    """
    Apply Median Smoothing filter (Non-linear filter).

    Theoretical basis:
        Replaces each pixel value with the median intensity of neighboring pixels
        within the sliding window. Extremely effective at removing impulse
        noise (Salt & Pepper noise) without blurring sharp edges.

    Args:
        image (np.ndarray): Input image.
        kernel_size (int): Aperture linear size. Must be an odd integer greater than 1.

    Returns:
        np.ndarray: Denoised image.
    """
    validate_image(image)

    if kernel_size <= 1:
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.medianBlur(image, kernel_size)


def sobel_edge_detection(
    image: np.ndarray,
    ksize: int = 3,
    direction: str = "both"
) -> np.ndarray:
    """
    Detect edges using the Sobel spatial gradient operator.

    Theoretical basis:
        Calculates first-order image derivatives using convolution masks:
            Gx = [[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]  (Horizontal changes / Vertical edges)
            Gy = [[-1, -2, -1], [ 0,  0,  0], [+1, +2, +1]]  (Vertical changes / Horizontal edges)
        Total gradient magnitude:
            G = sqrt(Gx^2 + Gy^2)

    Args:
        image (np.ndarray): Input image (grayscale or color).
        ksize (int): Size of extended Sobel kernel (1, 3, 5, or 7).
        direction (str): 'x' (horizontal gradient), 'y' (vertical gradient), or 'both'.

    Returns:
        np.ndarray: Gradient magnitude edge image (uint8 single-channel).
    """
    validate_image(image)

    # Sobel operator operates on single-channel grayscale
    gray = to_grayscale(image)

    # Ensure valid kernel size for cv2.Sobel (1, 3, 5, 7)
    if ksize not in [1, 3, 5, 7]:
        ksize = 3

    # Calculate gradients using 64-bit float to prevent underflow/overflow
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    if direction == "x":
        gradient = np.abs(sobel_x)
    elif direction == "y":
        gradient = np.abs(sobel_y)
    else:  # 'both'
        gradient = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Normalize gradient magnitude to [0, 255] uint8
    max_val = np.max(gradient)
    if max_val > 0:
        normalized = (gradient / max_val) * 255.0
    else:
        normalized = gradient

    return np.clip(normalized, 0, 255).astype(np.uint8)


def canny_edge_detection(
    image: np.ndarray,
    threshold1: float = 100.0,
    threshold2: float = 200.0
) -> np.ndarray:
    """
    Detect edges using the multi-stage Canny Edge Detector.

    Four fundamental stages of Canny algorithm:
    1. Noise Reduction: Smooth image with a 5x5 Gaussian filter.
    2. Intensity Gradient: Find gradient magnitude and direction with Sobel.
    3. Non-Maximum Suppression (NMS): Thin edges by removing spurious responses.
    4. Hysteresis Thresholding: Classify strong, weak, and non-edges using two thresholds.

    Args:
        image (np.ndarray): Input image.
        threshold1 (float): First threshold for hysteresis procedure (lower bound).
        threshold2 (float): Second threshold for hysteresis procedure (upper bound).

    Returns:
        np.ndarray: Binary edge map (uint8 single-channel, 0 or 255).
    """
    validate_image(image)

    gray = to_grayscale(image)

    # Ensure thresholds are in valid ordering
    t_lower = min(threshold1, threshold2)
    t_upper = max(threshold1, threshold2)

    return cv2.Canny(gray, threshold1=t_lower, threshold2=t_upper)


def binary_thresholding(
    image: np.ndarray,
    threshold_value: int = 127,
    max_val: int = 255,
    inverse: bool = False
) -> np.ndarray:
    """
    Segment image using global binary thresholding.

    Mathematical definition:
        dst(x, y) = max_val if src(x, y) > threshold_value else 0  (Standard)
        dst(x, y) = 0 if src(x, y) > threshold_value else max_val  (Inverse)

    Args:
        image (np.ndarray): Input image.
        threshold_value (int): Cutoff intensity (0-255).
        max_val (int): Output value for pixels exceeding threshold (default 255).
        inverse (bool): Whether to invert threshold logic.

    Returns:
        np.ndarray: Segmented binary image (uint8 single channel).
    """
    validate_image(image)
    gray = to_grayscale(image)

    thresh_type = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    _, thresholded = cv2.threshold(gray, threshold_value, max_val, thresh_type)

    return thresholded


def otsu_thresholding(
    image: np.ndarray,
    inverse: bool = False
) -> Tuple[np.ndarray, float]:
    """
    Segment image using Otsu's optimal automatic thresholding.

    Theoretical basis:
        Iterates across all possible threshold values (0-255) to find the one
        that maximizes between-class variance (sigma_b^2), which is equivalent
        to minimizing within-class variance of the two pixel populations (foreground/background).

    Args:
        image (np.ndarray): Input image.
        inverse (bool): Whether to invert binary polarity.

    Returns:
        Tuple[np.ndarray, float]: (Segmented binary image, Computed optimal threshold value).
    """
    validate_image(image)
    gray = to_grayscale(image)

    thresh_type = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    optimal_thresh, thresholded = cv2.threshold(
        gray, 0, 255, thresh_type + cv2.THRESH_OTSU
    )

    return thresholded, float(optimal_thresh)


def compute_histogram_data(image: np.ndarray) -> Dict[str, Any]:
    """
    Compute 256-bin intensity histograms for grayscale or BGR channels.

    Args:
        image (np.ndarray): Input image.

    Returns:
        Dict[str, Any]: Dictionary containing channel names and computed 1D frequency arrays.
    """
    validate_image(image)

    bins = np.arange(256)

    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        gray = image if len(image.shape) == 2 else image[:, :, 0]
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        return {
            "mode": "grayscale",
            "bins": bins,
            "channels": {"Grayscale (Intensity)": hist}
        }
    else:
        # BGR channels
        hist_b = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256]).flatten()
        hist_r = cv2.calcHist([image], [2], None, [256], [0, 256]).flatten()
        return {
            "mode": "color",
            "bins": bins,
            "channels": {
                "Blue Channel": (hist_b, "blue"),
                "Green Channel": (hist_g, "green"),
                "Red Channel": (hist_r, "red"),
            }
        }


def encode_image_to_png_bytes(image: np.ndarray) -> bytes:
    """
    Encode an OpenCV image array into PNG byte data for easy file downloading.

    Args:
        image (np.ndarray): Image array (uint8).

    Returns:
        bytes: Encoded PNG binary bytes.
    """
    validate_image(image)
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Failed to encode image to PNG format.")
    return buffer.tobytes()
