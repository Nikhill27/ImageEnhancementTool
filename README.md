# Image Enhancement and Filtering Tool Using Python and OpenCV

An academic mini-project developed for the **NMIMS B.Tech** course in **Image and Video Processing (IVP)**.

This application is built strictly using **classical digital image processing (DIP)** algorithms with **Python, OpenCV, NumPy, and Streamlit**. It does not rely on any machine learning models, deep learning networks, or cloud AI APIs.

---

## 📌 Project Overview

Digital image processing forms the backbone of computer vision, medical radiography, remote sensing, and industrial quality inspection. This interactive educational tool provides an intuitive graphical workbench where students, educators, and engineers can observe and experiment with foundational spatial-domain image transformations in real time.

---

## ✨ Features

1. **Image Ingestion**:
   - Upload any image in **JPG, JPEG, or PNG** format.
   - Built-in sample images provided (Low-contrast, Salt-and-pepper noise, Geometric shapes, Scenery) for instant demonstration.
2. **Point Transformations**:
   - **Grayscale Conversion**: Standard Rec.601 luminance weighting ($0.299R + 0.587G + 0.114B$).
   - **Digital Image Negative**: Intensity inversion ($s = 255 - r$).
3. **Contrast Enhancement & Histogram Processing**:
   - **Linear Contrast Stretching**: User-tunable or automatic $[r_{min}, r_{max}]$ clipping and normalization.
   - **Histogram Equalization (Global HE)**: Spreads intensity frequencies across the dynamic range. Preserves color fidelity by processing the luminance channel ($Y$) in $YCrCb$ color space.
4. **Spatial Neighborhood Smoothing (Low-Pass Filters)**:
   - **Gaussian Blur**: Weighted 2D spatial convolution with adjustable kernel size and standard deviation ($\sigma$).
   - **Median Filter**: Non-linear order-statistic filtering for removing impulse (salt-and-pepper) noise without edge degradation.
5. **Spatial Derivative & Edge Detection (High-Pass Filters)**:
   - **Sobel Operator**: Discrete first-order horizontal ($G_x$), vertical ($G_y$), and combined vector gradient magnitude ($\sqrt{G_x^2 + G_y^2}$).
   - **Canny Edge Detector**: 4-stage optimal edge detection with adjustable double-threshold hysteresis.
6. **Image Segmentation**:
   - **Global Binary Thresholding**: User-tunable threshold $T$ with optional polarity inversion.
   - **Otsu’s Thresholding**: Automatic maximization of inter-class variance to calculate optimal cutoff $T^*$.
7. **Visual Analytics & Export**:
   - Real-time side-by-side comparison of original and processed images.
   - Real-time Matplotlib intensity histogram curves (256 bins).
   - Clear explanations of algorithm concepts and mathematical formulas in plain English.
   - One-click **PNG Download** of the processed result.

---

## 📁 Project Structure

```text
IVP_Project/
│
├── app.py                   # Streamlit web user interface
├── image_processing.py      # Core classical image processing algorithms
├── generate_samples.py      # Utility script generating sample demo images
├── requirements.txt         # Project dependencies
├── README.md                # Setup, execution, and demonstration guide
├── report.md                # Academic mini-project documentation
│
├── sample_images/           # Sample test images for demonstration
│   ├── low_contrast.jpg
│   ├── salt_and_pepper_noise.png
│   ├── geometric_shapes.png
│   └── scenery.jpg
│
└── tests/
    └── test_processing.py  # Automated unit and integration test suite
```

---

## 🛠️ Prerequisites & Installation

### 1. Python Environment
Ensure **Python 3.9+** (tested on Python 3.12) is installed.

Verify by running in your terminal:
```bash
python --version
```

### 2. Clone or Navigate to Project Directory
```bash
cd c:\Users\Admin\OneDrive\Desktop\IVP_Project
```

### 3. Install Dependencies
Install all required libraries using pip:
```bash
python -m pip install -r requirements.txt
```

---

## 🚀 Running the Application

To launch the interactive Streamlit application, run:

```bash
python -m streamlit run app.py
```

Streamlit will start a local web server and automatically open the application in your default web browser (typically at `http://localhost:8501`).

---

## 🧪 Running Automated Tests

A comprehensive suite of 12 unit tests verifies every image processing function:

```bash
python -m pytest -v
```

Expected output:
```text
tests/test_processing.py::test_validate_image PASSED
tests/test_processing.py::test_grayscale_conversion PASSED
tests/test_processing.py::test_image_negative PASSED
tests/test_processing.py::test_contrast_stretching PASSED
tests/test_processing.py::test_histogram_equalization PASSED
tests/test_processing.py::test_gaussian_smoothing PASSED
tests/test_processing.py::test_median_smoothing PASSED
tests/test_processing.py::test_sobel_edge_detection PASSED
tests/test_processing.py::test_canny_edge_detection PASSED
tests/test_processing.py::test_binary_thresholding PASSED
tests/test_processing.py::test_otsu_thresholding PASSED
tests/test_processing.py::test_compute_histogram_and_png_encoding PASSED

============================== 12 passed in 0.47s ==============================
```

---

## 🎓 Step-by-Step Viva & Project Demonstration Guide

When presenting this project to your professor or examiner:

1. **Launch the App**: Run `python -m streamlit run app.py`.
2. **Demonstrate Contrast Enhancement**:
   - In the sidebar, select the sample image: `Low Contrast Scene`.
   - Apply **Contrast Stretching** and slide `r_min` and `r_max`. Show how the histogram widens.
   - Switch to **Histogram Equalization**. Show how the histogram flattens and features emerge clearly.
3. **Demonstrate Noise Reduction**:
   - Select the sample image: `Salt & Pepper Noise`.
   - Apply **Gaussian Smoothing** with kernel size 5. Point out that the noise remains partially visible while blurring details.
   - Switch to **Median Filtering** with kernel size 3 or 5. Show that salt-and-pepper noise is completely eliminated while boundaries stay crisp.
4. **Demonstrate Edge Detection**:
   - Select `Geometric Shapes`.
   - Apply **Sobel Edge Detection** and show `x`, `y`, and `both` gradient directions.
   - Switch to **Canny Edge Detection** and adjust the hysteresis sliders. Explain the four internal stages of Canny.
5. **Demonstrate Thresholding**:
   - Apply **Binary Thresholding** with manual slider.
   - Switch to **Otsu's Thresholding** and show the automatically calculated optimal threshold value $T^*$.
6. **Download Feature**:
   - Click the **⬇️ Download Processed Image** button to show direct export to PNG.

---

## 📚 NMIMS B.Tech IVP Syllabus Mapping

| Syllabus Unit | Course Topic | Implemented Function in `image_processing.py` |
| :--- | :--- | :--- |
| **Unit 1** | Digital Image Fundamentals | `to_grayscale()` |
| **Unit 2** | Spatial Domain Gray-level Slicing & Negatives | `image_negative()` |
| **Unit 2** | Piecewise-Linear Contrast Transformations | `contrast_stretching()` |
| **Unit 2** | Histogram Modeling & Equalization | `histogram_equalization()`, `compute_histogram_data()` |
| **Unit 3** | Smoothing Linear Spatial Filters | `gaussian_smoothing()` |
| **Unit 3** | Order-Statistic Non-linear Filters | `median_smoothing()` |
| **Unit 4** | Gradient Derivative Operators | `sobel_edge_detection()` |
| **Unit 4** | Multi-stage Edge Detection & Hysteresis | `canny_edge_detection()` |
| **Unit 5** | Intensity Thresholding Segmentation | `binary_thresholding()`, `otsu_thresholding()` |
