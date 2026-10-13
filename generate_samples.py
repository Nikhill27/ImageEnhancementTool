"""
Utility script to generate sample test images for the IVP project.
Provides ready-to-use images for demonstrating all mini-project features.
"""

import os
import numpy as np
import cv2


def generate_sample_images(output_dir="sample_images"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Low Contrast Image (Ideal for Contrast Stretching & Histogram Equalization)
    low_contrast = np.zeros((400, 500, 3), dtype=np.uint8)
    # Background in narrow range [100, 140]
    for y in range(400):
        for x in range(500):
            val = int(110 + 20 * np.sin(x / 40.0) + 10 * np.cos(y / 30.0))
            low_contrast[y, x] = [val - 5, val, val + 5]

    # Add shapes with slightly different intensity in narrow range
    cv2.circle(low_contrast, (250, 200), 90, (140, 145, 150), -1)
    cv2.rectangle(low_contrast, (80, 80), (180, 320), (95, 100, 105), -1)
    cv2.putText(low_contrast, "IVP Low Contrast", (110, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (125, 130, 135), 2)
    cv2.imwrite(os.path.join(output_dir, "low_contrast.jpg"), low_contrast)

    # 2. Salt and Pepper Noise Image (Ideal for Median vs Gaussian filter demo)
    clean_base = np.zeros((400, 500, 3), dtype=np.uint8)
    # Gradient backdrop
    for y in range(400):
        clean_base[y, :, 0] = int(200 - y * 0.3)
        clean_base[y, :, 1] = int(120 + y * 0.2)
        clean_base[y, :, 2] = int(80 + y * 0.3)

    cv2.circle(clean_base, (250, 200), 100, (255, 230, 50), -1)
    cv2.putText(clean_base, "Median Filter Demo", (80, 210), cv2.FONT_HERSHEY_DUPLEX, 1.0, (20, 20, 20), 2)

    noisy = clean_base.copy()
    num_noise = int(0.04 * 400 * 500)  # 4% noise
    # Add Pepper (black)
    for _ in range(num_noise // 2):
        ry = np.random.randint(0, 400)
        rx = np.random.randint(0, 500)
        noisy[ry, rx] = [0, 0, 0]
    # Add Salt (white)
    for _ in range(num_noise // 2):
        ry = np.random.randint(0, 400)
        rx = np.random.randint(0, 500)
        noisy[ry, rx] = [255, 255, 255]

    cv2.imwrite(os.path.join(output_dir, "salt_and_pepper_noise.png"), noisy)

    # 3. Geometric Shapes (Ideal for Edge Detection & Thresholding)
    shapes = np.full((400, 500, 3), 240, dtype=np.uint8)
    cv2.circle(shapes, (140, 150), 70, (40, 40, 200), -1)
    cv2.rectangle(shapes, (280, 80), (440, 220), (30, 160, 40), -1)
    pts = np.array([[250, 260], [150, 370], [350, 370]], np.int32)
    cv2.fillPoly(shapes, [pts], (200, 50, 40))
    cv2.putText(shapes, "Sobel & Canny", (150, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(output_dir, "geometric_shapes.png"), shapes)

    # 4. Colorful Scenery (Ideal for Grayscale, Negative, & General Enhancement)
    scenery = np.zeros((400, 500, 3), dtype=np.uint8)
    # Sky
    for y in range(250):
        scenery[y, :] = [int(220 - y * 0.4), int(160 - y * 0.2), 60]
    # Sun
    cv2.circle(scenery, (400, 80), 50, (30, 220, 255), -1)
    # Mountains
    mountain_pts1 = np.array([[0, 250], [140, 110], [280, 250]], np.int32)
    cv2.fillPoly(scenery, [mountain_pts1], (90, 80, 100))
    mountain_pts2 = np.array([[180, 250], [330, 130], [500, 250]], np.int32)
    cv2.fillPoly(scenery, [mountain_pts2], (70, 60, 80))
    # Ground
    for y in range(250, 400):
        scenery[y, :] = [40, int(130 + (y - 250) * 0.4), 30]

    cv2.imwrite(os.path.join(output_dir, "scenery.jpg"), scenery)
    print("Successfully generated all sample images in", output_dir)


if __name__ == "__main__":
    generate_sample_images()
