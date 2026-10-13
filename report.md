# ACADEMIC MINI-PROJECT REPORT

**Course:** Image and Video Processing (IVP)  
**Program:** B.Tech (Computer Engineering / Information Technology / Data Science)  
**Institution:** Mukesh Patel School of Technology Management & Engineering (MPSTME), NMIMS University  
**Academic Year:** 2026–2027  

---

# 1. Project Title

**Image Enhancement and Filtering Tool Using Python and OpenCV**

---

# 2. Abstract

In digital image processing, raw visual signals captured by digital cameras, sensors, and scanners frequently suffer from poor contrast, environmental illumination variances, sensor noise, and lack of distinct boundary definitions. Classical digital image processing (DIP) algorithms offer mathematically sound, deterministic, and computationally lightweight methodologies to enhance image quality and extract structural boundaries without the computational overhead or black-box nature of deep neural networks.

This academic mini-project presents the design, mathematical formulation, and software realization of an interactive desktop-web application entitled **"Image Enhancement and Filtering Tool"**. Developed using Python, OpenCV, NumPy, Matplotlib, and Streamlit, the application implements core algorithms aligned with the NMIMS B.Tech Image and Video Processing (IVP) syllabus: Grayscale Conversion, Digital Image Negative, Linear Contrast Stretching, Luminance-Preserving Histogram Equalization, 2D Gaussian Filtering, Non-linear Median Filtering, Sobel First-Order Gradient Edge Detection, Canny Multi-Stage Edge Detection, Global Binary Thresholding, and Otsu’s Optimal Inter-Class Variance Thresholding. The tool provides side-by-side visual comparisons, dynamic parameter tuning, real-time 256-bin histogram frequency plots, and one-click PNG image export. The implementation is verified through automated pytest test suites and evaluated using benchmark test images.

---

# 3. Introduction

Digital images represent discrete two-dimensional spatial arrays $f(x, y)$ of quantized intensity samples. Due to real-world physical constraints—including sensor thermal noise, atmospheric scattering, low light, and transmission imperfections—raw imagery often requires spatial enhancement before human observation or machine inspection.

Image enhancement operations are categorized broadly into:
1. **Point Transformations (Spatial Domain):** Operations where the output pixel value $s$ at coordinate $(x, y)$ depends solely on the input pixel value $r$ at the exact same spatial coordinate: $s = T(r)$. Examples include image negative and contrast stretching.
2. **Neighborhood / Spatial Filtering Operations:** Operations where the output value $g(x, y)$ depends on the pixel intensities within a defined spatial sub-window (kernel) centered at $(x, y)$. Examples include Gaussian smoothing, median filtering, and Sobel gradient convolution.
3. **Segmentation and Thresholding:** Segmenting the spatial matrix into foreground regions of interest and background regions based on intensity criteria.

This project bridges theoretical image processing mathematics taught in undergraduate university curricula with practical software implementation. By providing an interactive parameter manipulation interface, students and engineers can immediately observe the causal relationships between algorithmic parameters (e.g., standard deviation $\sigma$, kernel size $K$, threshold levels $T$) and output spatial characteristics.

---

# 4. Problem Statement

Undergraduate students studying Image and Video Processing often struggle to build an intuitive mental model connecting abstract mathematical equations (such as convolution sums, cumulative distribution functions, spatial gradient vectors, and between-class variance optimizations) with their concrete perceptual effects on real images. 

Furthermore, existing tools are either closed-source commercial platforms (e.g., MATLAB Image Processing Toolbox) that are heavy and license-restricted, or code-only scripting libraries (e.g., raw OpenCV scripts) lacking interactive visualization. There is a need for a lightweight, transparent, open-source, beginner-friendly academic platform that:
1. Realizes textbook classical image enhancement algorithms using standard scientific Python libraries.
2. Operates locally with zero cloud dependencies or machine learning overhead.
3. Exposes intuitive sliders and selectors for real-time comparative analysis.
4. Directly mirrors the university syllabus curriculum for laboratory demonstrations and examinations.

---

# 5. Objectives

The primary objectives of this mini-project are:
1. **Mathematical Implementation:** Implement classical digital image processing algorithms in clean, modular Python functions using OpenCV and NumPy.
2. **Interactive User Interface:** Develop an intuitive, modern Streamlit user interface featuring side-by-side image comparison and dynamic parameter controls.
3. **Statistical Histogram Visualization:** Compute and plot 256-bin intensity histograms comparing original and processed distributions.
4. **Educational Alignment:** Provide clear, beginner-friendly explanations, LaTeX equations, and syllabus references for every algorithm.
5. **Quality Assurance:** Validate algorithmic correctness and edge cases using an automated unit testing suite (`pytest`).
6. **Artifact Export:** Enable lossless downloading of processed images in PNG format.

---

# 6. Technologies Used

| Technology | Role | Justification |
| :--- | :--- | :--- |
| **Python 3.12** | Core Programming Language | Standard language for scientific computing, readable syntax, vast numerical ecosystem. |
| **OpenCV (`cv2`)** | Image Processing Engine | Highly optimized C++ backend with Python bindings for high-performance spatial convolution, color transformations, and gradient operations. |
| **NumPy** | Numerical Array Manipulation | High-performance $N$-dimensional array slicing, vectorized matrix arithmetic, and pixelwise clipping. |
| **Streamlit** | Graphical User Interface | Fast, reactive Python web framework facilitating sidebar parameter controls and side-by-side layout without complex HTML/JS boilerplate. |
| **Matplotlib** | Data Visualization | Precise plotting of 256-bin intensity histograms across grayscale and RGB color channels. |
| **Pillow (PIL)** | Image I/O Utility | File decoding and byte stream serialization. |
| **Pytest** | Test Automation Framework | Automated execution and regression testing of image transformation functions. |

---

# 7. System Architecture

The project follows a clean three-tier modular architecture:

```
+-------------------------------------------------------------+
|                     PRESENTATION TIER                       |
|          Streamlit Web Application Interface (app.py)       |
|  - Sidebar: Image Source Selector & Dynamic Sliders         |
|  - Main: Side-by-Side Image Previews & Stat Badges          |
|  - Analytics: Matplotlib Real-time 256-bin Histogram Plots  |
|  - Export: PNG Download Stream                              |
+------------------------------+------------------------------+
                               | Function Calls & Buffers
                               v
+-------------------------------------------------------------+
|                      LOGIC TIER                             |
|         Image Processing Engine (image_processing.py)       |
|  - Point Operations: to_grayscale(), image_negative()       |
|  - Contrast: contrast_stretching(), histogram_equalization()|
|  - Smoothing: gaussian_smoothing(), median_smoothing()      |
|  - Edges: sobel_edge_detection(), canny_edge_detection()   |
|  - Thresholding: binary_thresholding(), otsu_thresholding() |
|  - Utilities: compute_histogram_data(), png_encoding()      |
+------------------------------+------------------------------+
                               | Matrix Operations
                               v
+-------------------------------------------------------------+
|                  COMPUTATIONAL FOUNDATION                   |
|             OpenCV C++ Kernels & NumPy Arrays               |
+-------------------------------------------------------------+
```

### Data Flow Pipeline:
1. **Image Input:** An image is either uploaded by the user or chosen from built-in synthetic test patterns.
2. **Decoding:** The binary byte buffer is decoded into an 8-bit unsigned integer ($M \times N \times C$) NumPy matrix (`uint8`).
3. **Execution:** The selected algorithm function in `image_processing.py` is invoked with current parameter values.
4. **Rendering:** Both original and transformed arrays are rendered simultaneously in RGB space alongside statistical metrics (dimensions, channel count, min/max intensities).
5. **Histogram Plotting:** Both intensity distributions are rendered as line curves across 256 bins.
6. **Export:** The processed array is serialized to PNG byte stream for instant local download.

---

# 8. Explanation of Algorithms

### 8.1 Grayscale Conversion
- **Mathematical Principle:** Human visual perception has non-uniform spectral sensitivity to primary wavelengths (most sensitive to green, moderately to red, least to blue). Standard luminance conversion utilizes the ITU-R BT.601 standard:
  $$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
- **Implementation:** Vectorized matrix multiplication or `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`.
- **Purpose:** Compresses 3-channel data into a single 2D intensity array, reducing computational complexity by $66\%$ prior to edge detection or thresholding.

### 8.2 Digital Image Negative
- **Mathematical Principle:** Maps an intensity $r \in [0, L-1]$ to its inverted complement $s$:
  $$s = (L - 1) - r = 255 - r \quad (\text{for an 8-bit image where } L = 256)$$
- **Implementation:** Elementwise subtraction `255 - image`.
- **Purpose:** Enhances light details embedded within dark regions, especially useful in medical analysis (e.g., viewing bone hairline fractures in X-ray film).

### 8.3 Linear Contrast Stretching
- **Mathematical Principle:** Expands a narrow dynamic range $[r_{min}, r_{max}]$ to fill the full display spectrum $[0, 255]$:
  $$s = \left( \frac{r - r_{min}}{r_{max} - r_{min}} \right) \times 255$$
  Values below $r_{min}$ are clipped to 0; values above $r_{max}$ are clipped to 255.
- **Implementation:** Float conversion followed by vectorized scalar arithmetic, `np.clip()`, and casting back to `uint8`.
- **Purpose:** Restores clarity to poorly lit or overexposed images without altering underlying spatial relationships.

### 8.4 Histogram Equalization
- **Mathematical Principle:** For a discrete grayscale image of dimensions $M \times N$, let $n_k$ be the number of pixels with intensity $r_k$. The probability of occurrence is:
  $$p_r(r_k) = \frac{n_k}{M \cdot N}, \quad k = 0, 1, \dots, L - 1$$
  The transformation function based on the Cumulative Distribution Function (CDF) is:
  $$s_k = T(r_k) = (L - 1) \sum_{j=0}^{k} p_r(r_j)$$
- **Color Preservation Technique:** Directly equalizing individual R, G, and B color channels causes unacceptable hue shifts. This tool transforms the BGR image to the $YCrCb$ color space, applies histogram equalization exclusively to the luminance channel $Y$, and converts back to BGR. Chromaticity ($Cr, Cb$) is perfectly preserved.

### 8.5 Gaussian Smoothing Filter
- **Mathematical Principle:** A linear low-pass convolution operator defined by the continuous 2D Gaussian function:
  $$G(x, y) = \frac{1}{2\pi \sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$
  The filter assigns maximum weight to the central pixel, with weights decaying smoothly as radial distance increases according to standard deviation $\sigma$.
- **Implementation:** `cv2.GaussianBlur(image, (k, k), sigma)`.
- **Purpose:** Suppresses high-frequency sensor noise while preserving edge boundaries more effectively than a standard box filter.

### 8.6 Median Smoothing Filter
- **Mathematical Principle:** A non-linear order-statistic spatial filter. Within a sliding neighborhood window $S_{xy}$ centered at $(x, y)$:
  $$g(x, y) = \text{median}\left(\{f(s, t) \mid (s, t) \in S_{xy}\}\right)$$
- **Implementation:** `cv2.medianBlur(image, k)`.
- **Key Advantage:** Unlike linear averaging (which smears extreme noise pixels into surrounding areas), the median filter completely discards impulse noise outliers (salt-and-pepper noise) while preserving crisp, step-like edge boundaries.

### 8.7 Sobel Edge Detection
- **Mathematical Principle:** Computes discrete first-order spatial derivatives using pair-wise $3 \times 3$ convolution masks:
  $$G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix} * f, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix} * f$$
  Total gradient magnitude:
  $$G = \sqrt{G_x^2 + G_y^2}$$
- **Implementation:** Computed using 64-bit floating point buffers to prevent arithmetic overflow, followed by normalization to $[0, 255]$ `uint8`.

### 8.8 Canny Edge Detection
- **Mathematical Principle:** Recognized as the optimal edge detector based on three criteria:
  1. Low error rate (marking only true edges).
  2. Edge localization (marked edge points match real edge center).
  3. Single response (no duplicate responses for a single edge).
- **Four Algorithmic Stages:**
  1. *Gaussian Filter:* Smooths the image to eliminate noise.
  2. *Gradient Calculation:* Computes gradient magnitude and direction ($\theta = \arctan(G_y / G_x)$).
  3. *Non-Maximum Suppression (NMS):* Thins edges to 1-pixel width by suppressing pixels that are not local maxima along the gradient direction.
  4. *Hysteresis Thresholding:* Uses two thresholds ($T_{low}, T_{high}$). Pixels with $G > T_{high}$ are strong edges; pixels with $T_{low} \le G \le T_{high}$ are preserved only if connected to strong edges.

### 8.9 Global Binary Thresholding
- **Mathematical Principle:** Segments an image into binary partitions (foreground and background) using a fixed intensity cutoff $T$:
  $$g(x, y) = \begin{cases} 255 & \text{if } f(x, y) > T \\ 0 & \text{otherwise} \end{cases}$$
- **Implementation:** `cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)`.

### 8.10 Otsu's Automatic Thresholding
- **Mathematical Principle:** Eliminates manual threshold guessing by statistically determining the optimal threshold $T^*$ that maximizes between-class variance $\sigma_B^2$:
  $$\sigma_B^2(t) = \omega_0(t) \omega_1(t) \left[\mu_0(t) - \mu_1(t)\right]^2$$
  Where:
  - $\omega_0(t), \omega_1(t)$: Cumulative probabilities of the background and foreground classes.
  - $\mu_0(t), \mu_1(t)$: Mean intensity values of the background and foreground classes.
  The algorithm exhaustively evaluates $t \in [0, 255]$ and selects $T^* = \arg\max_t \sigma_B^2(t)$.

---

# 9. Methodology

The development process followed an iterative software engineering lifecycle tailored for academic mini-projects:

1. **Curriculum Decomposition:** Analyzed the NMIMS B.Tech IVP syllabus modules to identify fundamental spatial and intensity algorithms.
2. **Core Algorithm Implementation:** Designed `image_processing.py` containing modular, pure Python functions with clear type hints, docstrings, and input validation.
3. **Synthetic Test Data Generation:** Created `generate_samples.py` to synthesize four distinct benchmark images tailored to demonstrate specific algorithms (e.g., low-contrast scenes, impulse noise, geometric contours).
4. **GUI Construction:** Developed `app.py` utilizing Streamlit layout primitives (`st.columns`, `st.sidebar`, `st.expander`, `st.download_button`).
5. **Histogram Integration:** Implemented 256-bin histogram computation using `cv2.calcHist` and plotted with Matplotlib.
6. **Automated Verification:** Authored unit tests in `tests/test_processing.py` covering mathematical correctness, boundary edge cases, and invalid inputs using `pytest`.

---

# 10. Expected Results and Testing Procedure

### Automated Test Results
All functions were verified using `pytest`. The test suite covers 12 dedicated test cases:

```text
============================= test session starts =============================
platform win32 -- Python 3.12.6, pytest-9.1.1
rootdir: C:\Users\Admin\OneDrive\Desktop\IVP_Project
collected 12 items

tests/test_processing.py::test_validate_image PASSED                     [  8%]
tests/test_processing.py::test_grayscale_conversion PASSED               [ 16%]
tests/test_processing.py::test_image_negative PASSED                     [ 25%]
tests/test_processing.py::test_contrast_stretching PASSED                [ 33%]
tests/test_processing.py::test_histogram_equalization PASSED             [ 41%]
tests/test_processing.py::test_gaussian_smoothing PASSED                 [ 50%]
tests/test_processing.py::test_median_smoothing PASSED                   [ 58%]
tests/test_processing.py::test_sobel_edge_detection PASSED               [ 66%]
tests/test_processing.py::test_canny_edge_detection PASSED               [ 75%]
tests/test_processing.py::test_binary_thresholding PASSED                [ 83%]
tests/test_processing.py::test_otsu_thresholding PASSED                  [ 91%]
tests/test_processing.py::test_compute_histogram_and_png_encoding PASSED [100%]

============================== 12 passed in 0.47s ==============================
```

### Visual Verification Summary:
- **Contrast Stretching:** Successfully stretched synthetic low-contrast images (initial range $[95, 150]$) to fill $[0, 255]$.
- **Histogram Equalization:** The intensity histogram flattened noticeably; hidden details in low-contrast scenes became prominently visible.
- **Median vs. Gaussian Filter on Salt-and-Pepper Noise:** Gaussian filtering smeared the white and black speckles into blurry blotches, whereas median filtering completely removed the speckles while keeping object borders sharp.
- **Sobel vs. Canny on Shapes:** Sobel produced thick, gradient-proportional boundaries; Canny produced thin, continuous 1-pixel-wide contours through non-maximum suppression and hysteresis.
- **Otsu's Thresholding:** Correctly computed the bimodal valley threshold and segmented foreground objects cleanly without manual intervention.

---

# 11. Applications

1. **Medical Imaging Diagnostics:** Enhancing contrast in X-rays, mammograms, and MRI scans; negative transformation aids in micro-calcification detection.
2. **Document Image Processing & OCR:** Binarizing historical manuscripts, scanned receipts, and book pages using Otsu's thresholding before text recognition.
3. **Autonomous Navigation & Robotics:** Extracting road lanes, boundaries, and obstacles using Canny edge detection and Gaussian smoothing.
4. **Satellite & Remote Sensing:** Equalizing atmospheric haze in satellite imagery to analyze vegetation indices and geographical topography.
5. **Industrial Quality Control:** Inspecting printed circuit boards (PCBs) for manufacturing defects, missing solder joints, or crack propagation.

---

# 12. Connection to NMIMS Image and Video Processing Syllabus

This project directly aligns with the **B.Tech Image and Video Processing (IVP)** syllabus prescribed by **NMIMS University**:

| Syllabus Unit | Course Topic | Project Realization |
| :--- | :--- | :--- |
| **Unit 1: Fundamentals of Digital Image Processing** | Image sampling, quantization, color models (RGB, YCrCb, Grayscale), matrix representations. | Implemented in `to_grayscale()` and color-space conversions (`cv2.cvtColor`, `cv2.split`). |
| **Unit 2: Spatial Domain Transformations** | Point processing, digital negative, contrast stretching, gray-level slicing. | Implemented in `image_negative()` and `contrast_stretching()`. |
| **Unit 2: Histogram Processing** | Histogram computation, equalization, contrast enhancement. | Implemented in `histogram_equalization()` and real-time dual-histogram visualization in `app.py`. |
| **Unit 3: Spatial Filtering Techniques** | Linear smoothing filters (Gaussian), non-linear order-statistic filters (Median). | Implemented in `gaussian_smoothing()` and `median_smoothing()`. |
| **Unit 4: Image Edge Detection** | First-order gradient operators (Sobel), multi-stage optimal edge detectors (Canny). | Implemented in `sobel_edge_detection()` and `canny_edge_detection()`. |
| **Unit 5: Image Segmentation** | Global thresholding, Otsu's optimal variance thresholding. | Implemented in `binary_thresholding()` and `otsu_thresholding()`. |

---

# 13. Conclusion

The **Image Enhancement and Filtering Tool** successfully demonstrates the implementation and visual effect of foundational classical digital image processing algorithms. By combining Python's scientific ecosystem (OpenCV and NumPy) with the modern Streamlit framework, the project delivers an interactive, responsive, and educationally rich workbench.

The tool fulfills all academic objectives: it avoids opaque machine learning models, exposes mathematical formulations and parameters transparently, provides real-time dual histogram analytics, and verifies all transformations via automated test suites. It serves as an exemplary academic mini-project suitable for university laboratory evaluations, vivas, and classroom demonstrations.

---

# 14. References

1. **Gonzalez, R. C., & Woods, R. E.** (2018). *Digital Image Processing* (4th ed.). Pearson Education.
2. **Jain, A. K.** (1989). *Fundamentals of Digital Image Processing*. Prentice-Hall.
3. **Canny, J.** (1986). "A Computational Approach to Edge Detection". *IEEE Transactions on Pattern Analysis and Machine Intelligence*, PAMI-8(6), 679–698.
4. **Otsu, N.** (1979). "A Threshold Selection Method from Gray-Level Histograms". *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), 62–66.
5. **Bradski, G.** (2000). "The OpenCV Library". *Dr. Dobb's Journal of Software Tools*.
6. **Streamlit Documentation** (2026). *Streamlit: An open-source Python framework for machine learning and data science*. https://docs.streamlit.io
