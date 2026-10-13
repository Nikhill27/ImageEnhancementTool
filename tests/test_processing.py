"""
Unit and Integration Tests for Image Processing Algorithms
NMIMS B.Tech - Image and Video Processing (IVP)

Tests all classical digital image processing routines in image_processing.py:
- Grayscale conversion
- Digital image negative
- Linear contrast stretching
- Histogram equalization (Grayscale and Color)
- Gaussian smoothing and Median filtering
- Sobel and Canny edge detection
- Binary and Otsu's thresholding
- Histogram computation and PNG encoding
- Edge cases and error handling
"""

import pytest
import numpy as np
import cv2
import image_processing as ip


@pytest.fixture
def synthetic_color_image():
    """Create a 100x100 3-channel synthetic BGR test image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Top-left blue, top-right green, bottom-left red, bottom-right white
    img[:50, :50] = [255, 0, 0]      # Blue
    img[:50, 50:] = [0, 255, 0]      # Green
    img[50:, :50] = [0, 0, 255]      # Red
    img[50:, 50:] = [255, 255, 255]  # White
    return img


@pytest.fixture
def synthetic_gray_image():
    """Create a 100x100 single-channel gradient grayscale test image."""
    x = np.linspace(50, 200, 100, dtype=np.uint8)
    return np.tile(x, (100, 1))


def test_validate_image():
    """Test validation handles invalid and empty inputs gracefully."""
    with pytest.raises(ValueError, match="non-empty NumPy array"):
        ip.validate_image(None)

    with pytest.raises(ValueError, match="empty"):
        ip.validate_image(np.array([]))


def test_grayscale_conversion(synthetic_color_image, synthetic_gray_image):
    """Test conversion of 3-channel color image to single-channel grayscale."""
    gray = ip.to_grayscale(synthetic_color_image)
    assert gray.ndim == 2
    assert gray.shape == (100, 100)
    assert gray.dtype == np.uint8

    # Grayscale image passed to to_grayscale should return a copy of itself
    gray_already = ip.to_grayscale(synthetic_gray_image)
    assert np.array_equal(gray_already, synthetic_gray_image)


def test_image_negative(synthetic_gray_image):
    """Test digital negative transformation s = 255 - r."""
    negative = ip.image_negative(synthetic_gray_image)
    assert negative.shape == synthetic_gray_image.shape
    # Check mathematical inversion
    expected = 255 - synthetic_gray_image
    assert np.array_equal(negative, expected)
    # Double negative should return original
    assert np.array_equal(ip.image_negative(negative), synthetic_gray_image)


def test_contrast_stretching(synthetic_gray_image):
    """Test min-max contrast stretching expands the dynamic range."""
    stretched = ip.contrast_stretching(synthetic_gray_image, r_min=50, r_max=200)
    assert stretched.min() == 0
    assert stretched.max() == 255
    assert stretched.shape == synthetic_gray_image.shape

    # Test edge case: uniform flat image (prevent division by zero)
    flat = np.full((50, 50), 128, dtype=np.uint8)
    stretched_flat = ip.contrast_stretching(flat)
    assert stretched_flat.shape == (50, 50)


def test_histogram_equalization(synthetic_color_image, synthetic_gray_image):
    """Test histogram equalization on both single-channel and multi-channel images."""
    # Grayscale HE
    eq_gray = ip.histogram_equalization(synthetic_gray_image)
    assert eq_gray.shape == synthetic_gray_image.shape
    assert eq_gray.dtype == np.uint8

    # Color HE (preserves 3 channels via YCrCb)
    eq_color = ip.histogram_equalization(synthetic_color_image)
    assert eq_color.shape == synthetic_color_image.shape
    assert eq_color.dtype == np.uint8


def test_gaussian_smoothing(synthetic_color_image):
    """Test Gaussian smoothing filter with odd and even kernel inputs."""
    # Standard odd kernel
    smoothed = ip.gaussian_smoothing(synthetic_color_image, kernel_size=5, sigma=1.2)
    assert smoothed.shape == synthetic_color_image.shape

    # Automatic correction of even kernel to odd
    smoothed_even = ip.gaussian_smoothing(synthetic_color_image, kernel_size=4, sigma=1.0)
    assert smoothed_even.shape == synthetic_color_image.shape


def test_median_smoothing():
    """Test median filter effectively eliminates salt-and-pepper noise."""
    # Create image with artificial salt-and-pepper noise
    img = np.full((50, 50), 120, dtype=np.uint8)
    img[10, 10] = 0    # Pepper
    img[20, 20] = 255  # Salt

    denoised = ip.median_smoothing(img, kernel_size=3)
    # The isolated noise pixels should be restored to surrounding median (120)
    assert denoised[10, 10] == 120
    assert denoised[20, 20] == 120


def test_sobel_edge_detection():
    """Test Sobel operator detects sharp intensity transitions."""
    # Vertical step edge: left is black (0), right is white (255)
    step_edge = np.zeros((60, 60), dtype=np.uint8)
    step_edge[:, 30:] = 255

    sobel_both = ip.sobel_edge_detection(step_edge, ksize=3, direction="both")
    sobel_x = ip.sobel_edge_detection(step_edge, ksize=3, direction="x")
    sobel_y = ip.sobel_edge_detection(step_edge, ksize=3, direction="y")

    # Horizontal derivative (vertical edge) should have high response along column 29-31
    assert np.max(sobel_x[:, 29:32]) > 200
    # Vertical derivative across vertical edge should be near zero
    assert np.max(sobel_y) < 10
    # Combined gradient magnitude should be strong
    assert np.max(sobel_both) > 200


def test_canny_edge_detection():
    """Test Canny detector generates binary edge map (0 and 255 values)."""
    step_edge = np.zeros((60, 60), dtype=np.uint8)
    step_edge[:, 30:] = 255

    edges = ip.canny_edge_detection(step_edge, threshold1=50, threshold2=150)
    assert edges.shape == (60, 60)
    # Unique values should only be 0 (background) and 255 (edge)
    unique_vals = set(np.unique(edges))
    assert unique_vals.issubset({0, 255})
    # Edge detected near boundary
    assert 255 in unique_vals


def test_binary_thresholding(synthetic_gray_image):
    """Test global binary thresholding segmentation."""
    thresh = ip.binary_thresholding(synthetic_gray_image, threshold_value=127)
    assert thresh.shape == (100, 100)
    assert set(np.unique(thresh)).issubset({0, 255})

    # Inverted threshold
    thresh_inv = ip.binary_thresholding(synthetic_gray_image, threshold_value=127, inverse=True)
    assert np.array_equal(thresh, 255 - thresh_inv)


def test_otsu_thresholding(synthetic_gray_image):
    """Test Otsu's optimal automatic thresholding."""
    thresh, optimal_t = ip.otsu_thresholding(synthetic_gray_image)
    assert thresh.shape == (100, 100)
    assert 0 <= optimal_t <= 255
    assert set(np.unique(thresh)).issubset({0, 255})


def test_compute_histogram_and_png_encoding(synthetic_color_image, synthetic_gray_image):
    """Test histogram computation and PNG byte conversion."""
    hist_gray = ip.compute_histogram_data(synthetic_gray_image)
    assert hist_gray["mode"] == "grayscale"
    assert len(hist_gray["bins"]) == 256

    hist_color = ip.compute_histogram_data(synthetic_color_image)
    assert hist_color["mode"] == "color"
    assert "Red Channel" in hist_color["channels"]

    png_bytes = ip.encode_image_to_png_bytes(synthetic_color_image)
    assert isinstance(png_bytes, bytes)
    # Standard PNG file magic signature: \x89PNG\r\n\x1a\n
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
