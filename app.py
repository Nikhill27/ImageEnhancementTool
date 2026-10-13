"""
Streamlit Web Application: Image Enhancement and Filtering Tool
NMIMS B.Tech - Image and Video Processing (IVP) Academic Mini-Project

Developed strictly using classical digital image processing principles
(OpenCV, NumPy, Matplotlib, Streamlit) without any deep learning or AI APIs.
"""

import os
from io import BytesIO
from typing import Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

import image_processing as ip

# Page configuration
st.set_page_config(
    page_title="Image Enhancement & Filtering Tool | NMIMS IVP",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished academic UI
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #DBEAFE;
        color: #1E40AF;
        margin-right: 0.4rem;
    }
    .info-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 1rem 1.25rem;
        border-radius: 0.375rem;
        margin-bottom: 1.5rem;
    }
    .stat-box {
        background: #F1F5F9;
        padding: 0.6rem 0.8rem;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Educational Descriptions for each algorithm
ALGORITHM_DETAILS = {
    "Grayscale Conversion": {
        "title": "Grayscale Conversion",
        "description": "Converts a multi-channel color (RGB/BGR) image into a single-channel intensity image by computing weighted luminance across channels.",
        "formula": "Y = 0.299 \\cdot R + 0.587 \\cdot G + 0.114 \\cdot B",
        "syllabus_unit": "Unit 1: Digital Image Fundamentals & Color Representations",
        "use_case": "Essential preprocessing step for dimensionality reduction before edge detection, feature extraction, and segmentation.",
    },
    "Image Negative": {
        "title": "Digital Image Negative",
        "description": "Inverts pixel intensity levels. Light details stand out in predominantly dark scenes and vice versa, mimicking photographic film negatives.",
        "formula": "s = (L - 1) - r = 255 - r \\quad \\text{for 8-bit images}",
        "syllabus_unit": "Unit 2: Spatial Domain Intensity Transformations",
        "use_case": "Enhancing white or gray details embedded in black regions, widely used in medical imaging (e.g., X-rays, mammograms).",
    },
    "Contrast Stretching": {
        "title": "Linear Contrast Stretching (Normalization)",
        "description": "Expands a narrow dynamic range of gray levels to span the full [0, 255] spectrum using piecewise linear transformation.",
        "formula": "s = \\frac{r - r_{min}}{r_{max} - r_{min}} \\times 255",
        "syllabus_unit": "Unit 2: Image Enhancement in the Spatial Domain",
        "use_case": "Corrects washed out, poorly illuminated, or overexposed images with concentrated intensity distributions.",
    },
    "Histogram Equalization": {
        "title": "Histogram Equalization (Global HE)",
        "description": "Flattens the probability density function (PDF) of image intensities, spreading out the most frequent intensity values to yield a uniform histogram and increased dynamic contrast.",
        "formula": "s_k = T(r_k) = (L - 1) \\sum_{j=0}^{k} p_r(r_j) = \\frac{L - 1}{MN} \\sum_{j=0}^{k} n_j",
        "syllabus_unit": "Unit 2: Histogram Processing & Enhancement",
        "use_case": "Medical radiography, satellite imagery, and forensic analysis where lighting cannot be controlled.",
    },
    "Gaussian Smoothing": {
        "title": "Gaussian Smoothing Filter",
        "description": "A 2D spatial convolution low-pass filter with weights derived from the Gaussian distribution. Pixels closer to the center receive higher weights, reducing high-frequency noise with minimal blur.",
        "formula": "G(x, y) = \\frac{1}{2\\pi \\sigma^2} \\exp\\left(-\\frac{x^2 + y^2}{2\\sigma^2}\\right)",
        "syllabus_unit": "Unit 3: Spatial Filtering & Image Smoothing",
        "use_case": "Suppressing Gaussian sensor noise and anti-aliasing prior to edge detection operations.",
    },
    "Median Filtering": {
        "title": "Median Smoothing Filter",
        "description": "A non-linear spatial neighborhood filter that replaces each pixel with the statistical median of neighboring intensities. Unlike linear blurring, it completely rejects extreme impulse outliers without softening sharp edges.",
        "formula": "g(x, y) = \\text{median}\\{f(s, t) \\mid (s, t) \\in S_{xy}\\}",
        "syllabus_unit": "Unit 3: Non-linear Order-Statistic Filtering",
        "use_case": "Eliminating salt-and-pepper noise caused by faulty sensor pixels, transmission errors, or ADC errors.",
    },
    "Sobel Edge Detection": {
        "title": "Sobel Spatial Gradient Operator",
        "description": "Calculates discrete first-order spatial derivatives along horizontal (Gx) and vertical (Gy) directions using 3x3 convolution kernels, followed by vector gradient magnitude computation.",
        "formula": "G = \\sqrt{G_x^2 + G_y^2}, \\quad G_x = \\begin{bmatrix} -1 & 0 & 1 \\\\ -2 & 0 & 2 \\\\ -1 & 0 & 1 \\end{bmatrix} * I, \\quad G_y = \\begin{bmatrix} -1 & -2 & -1 \\\\ 0 & 0 & 0 \\\\ 1 & 2 & 1 \\end{bmatrix} * I",
        "syllabus_unit": "Unit 4: First-Order Derivative Operators & Edge Detection",
        "use_case": "Boundary extraction, shape analysis, feature segmentation, and autonomous vision systems.",
    },
    "Canny Edge Detection": {
        "title": "Canny Multi-Stage Edge Detector",
        "description": "An optimal edge detector based on three criteria (low error rate, localized edges, single edge response). Involves Gaussian smoothing, gradient computation, non-maximum suppression (thinning), and double-threshold hysteresis.",
        "formula": "\\text{Stages: } 1. \\text{ Gaussian Blur} \\rightarrow 2. \\text{ Sobel Gradient} \\rightarrow 3. \\text{ Non-Max Suppression} \\rightarrow 4. \\text{ Hysteresis Thresholding}",
        "syllabus_unit": "Unit 4: Advanced Edge Detection Algorithms",
        "use_case": "Standard industry benchmark for object detection, lane detection, robotics, and medical boundary identification.",
    },
    "Binary Thresholding": {
        "title": "Global Binary Thresholding",
        "description": "Converts a grayscale image into a binary black-and-white mask by comparing every pixel against a fixed threshold cutoff value T.",
        "formula": "g(x, y) = \\begin{cases} 255 & \\text{if } f(x, y) > T \\\\ 0 & \\text{otherwise} \\end{cases}",
        "syllabus_unit": "Unit 5: Image Segmentation & Thresholding",
        "use_case": "Optical character recognition (OCR), document scanning, and separating foreground objects from bright backgrounds.",
    },
    "Otsu's Thresholding": {
        "title": "Otsu's Automatic Thresholding",
        "description": "Automatically calculates an optimal global threshold by maximizing inter-class variance (between foreground and background classes) or equivalently minimizing intra-class variance.",
        "formula": "\\sigma_B^2(t) = \\omega_0(t) \\omega_1(t) \\left[\\mu_0(t) - \\mu_1(t)\\right]^2 \\quad \\rightarrow \\quad T^* = \\arg\\max_t \\sigma_B^2(t)",
        "syllabus_unit": "Unit 5: Optimal Automatic Thresholding Algorithms",
        "use_case": "Automated document binarization, industrial inspection, and automated cell segmentation in microscopy.",
    },
}


def load_image_from_upload(uploaded_file) -> Optional[np.ndarray]:
    """Load an uploaded image into a NumPy BGR image array."""
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img_bgr
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None


def plot_histograms(orig_bgr: np.ndarray, proc_img: np.ndarray) -> plt.Figure:
    """Generate Matplotlib figure comparing histograms of original and processed images."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=100)
    fig.patch.set_facecolor("#FFFFFF")

    # Original Image Histogram
    ax_orig = axes[0]
    ax_orig.set_title("Original Image Histogram", fontsize=11, fontweight="bold", pad=8)
    ax_orig.set_xlabel("Pixel Intensity [0 - 255]", fontsize=9)
    ax_orig.set_ylabel("Pixel Frequency", fontsize=9)
    ax_orig.set_xlim([0, 256])
    ax_orig.grid(True, linestyle="--", alpha=0.4)

    orig_data = ip.compute_histogram_data(orig_bgr)
    if orig_data["mode"] == "grayscale":
        ax_orig.plot(orig_data["bins"], orig_data["channels"]["Grayscale (Intensity)"], color="#2B2D42", lw=1.5)
    else:
        for ch_name, (hist, color) in orig_data["channels"].items():
            ax_orig.plot(orig_data["bins"], hist, color=color, label=ch_name, lw=1.2, alpha=0.85)
        ax_orig.legend(loc="upper right", fontsize=8)

    # Processed Image Histogram
    ax_proc = axes[1]
    ax_proc.set_title("Processed Image Histogram", fontsize=11, fontweight="bold", pad=8)
    ax_proc.set_xlabel("Pixel Intensity [0 - 255]", fontsize=9)
    ax_proc.set_ylabel("Pixel Frequency", fontsize=9)
    ax_proc.set_xlim([0, 256])
    ax_proc.grid(True, linestyle="--", alpha=0.4)

    proc_data = ip.compute_histogram_data(proc_img)
    if proc_data["mode"] == "grayscale":
        ax_proc.plot(proc_data["bins"], proc_data["channels"]["Grayscale (Intensity)"], color="#9333EA", lw=1.5)
    else:
        for ch_name, (hist, color) in proc_data["channels"].items():
            ax_proc.plot(proc_data["bins"], hist, color=color, label=ch_name, lw=1.2, alpha=0.85)
        ax_proc.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig


def main():
    # Header Banner
    st.markdown('<div class="main-header">Image Enhancement & Filtering Tool</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sub-header">
            <span class="badge">NMIMS B.Tech IVP</span>
            <span class="badge">OpenCV & NumPy</span>
            <span class="badge">Classical DIP</span>
            <span>Interactive academic workbench for digital image enhancement, spatial filtering, edge detection, and thresholding.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar: Input Image Selection
    st.sidebar.header("1. Input Image")
    input_source = st.sidebar.radio(
        "Choose Image Source:",
        ("Upload Image (JPG / PNG)", "Select Built-in Sample Image"),
        index=1,
    )

    orig_bgr = None
    image_name = "sample"

    if input_source == "Upload Image (JPG / PNG)":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Image file:",
            type=["jpg", "jpeg", "png"],
            help="Supports JPG, JPEG, and PNG formats.",
        )
        if uploaded_file is not None:
            orig_bgr = load_image_from_upload(uploaded_file)
            image_name = uploaded_file.name.rsplit(".", 1)[0]
        else:
            st.sidebar.info("Please upload an image or switch to a built-in sample image.")
    else:
        # Sample images
        sample_options = {
            "Low Contrast Scene (Best for Stretching & Equalization)": "sample_images/low_contrast.jpg",
            "Salt & Pepper Noise (Best for Median Filtering)": "sample_images/salt_and_pepper_noise.png",
            "Geometric Shapes (Best for Sobel, Canny & Thresholding)": "sample_images/geometric_shapes.png",
            "Colorful Scenery (Best for Grayscale, Negative & Blur)": "sample_images/scenery.jpg",
        }
        chosen_sample = st.sidebar.selectbox("Choose Sample Image:", list(sample_options.keys()))
        sample_path = sample_options[chosen_sample]

        # Auto-generate sample images if not yet present
        if not os.path.exists(sample_path):
            import generate_samples
            generate_samples.generate_sample_images()

        if os.path.exists(sample_path):
            orig_bgr = cv2.imread(sample_path)
            image_name = os.path.basename(sample_path).rsplit(".", 1)[0]
        else:
            st.sidebar.error("Sample image could not be loaded.")

    if orig_bgr is None:
        st.warning("⚠️ No image loaded. Please select or upload an image from the sidebar to begin.")
        st.stop()

    # Sidebar: Operation Selection
    st.sidebar.markdown("---")
    st.sidebar.header("2. Processing Operation")
    operation = st.sidebar.selectbox(
        "Select Algorithm to Apply:",
        [
            "Grayscale Conversion",
            "Image Negative",
            "Contrast Stretching",
            "Histogram Equalization",
            "Gaussian Smoothing",
            "Median Filtering",
            "Sobel Edge Detection",
            "Canny Edge Detection",
            "Binary Thresholding",
            "Otsu's Thresholding",
        ],
    )

    # Sidebar: Dynamic Parameters
    st.sidebar.markdown("---")
    st.sidebar.header("3. Algorithm Parameters")

    processed_img = None
    computed_meta = {}

    try:
        if operation == "Grayscale Conversion":
            st.sidebar.caption("Converts BGR color representation to single-channel 8-bit luminance.")
            processed_img = ip.to_grayscale(orig_bgr)

        elif operation == "Image Negative":
            st.sidebar.caption("Applies intensity inversion: s = 255 - r.")
            processed_img = ip.image_negative(orig_bgr)

        elif operation == "Contrast Stretching":
            st.sidebar.caption("Adjust lower (r_min) and upper (r_max) clipping bounds.")
            auto_bounds = st.sidebar.checkbox("Auto Min-Max (Based on current image)", value=False)
            if auto_bounds:
                cur_min = int(np.min(orig_bgr))
                cur_max = int(np.max(orig_bgr))
                r_min = st.sidebar.slider("r_min (Lower Threshold)", 0, 255, cur_min)
                r_max = st.sidebar.slider("r_max (Upper Threshold)", 0, 255, cur_max)
            else:
                r_min = st.sidebar.slider("r_min (Lower Threshold)", 0, 255, 30)
                r_max = st.sidebar.slider("r_max (Upper Threshold)", 0, 255, 220)

            if r_min >= r_max:
                st.sidebar.warning("Note: r_min must be less than r_max. Adjusting automatically.")
                r_max = min(255, r_min + 1)

            processed_img = ip.contrast_stretching(orig_bgr, r_min=r_min, r_max=r_max)
            computed_meta["Bounds"] = f"[{r_min}, {r_max}]"

        elif operation == "Histogram Equalization":
            st.sidebar.caption("Applies global histogram equalization (luminance channel for color).")
            processed_img = ip.histogram_equalization(orig_bgr)

        elif operation == "Gaussian Smoothing":
            st.sidebar.caption("Low-pass linear Gaussian filtering.")
            k_size = st.sidebar.slider("Kernel Size (pixels)", 3, 31, 5, step=2)
            sigma = st.sidebar.slider("Sigma (Standard Deviation)", 0.2, 10.0, 1.5, step=0.1)
            processed_img = ip.gaussian_smoothing(orig_bgr, kernel_size=k_size, sigma=sigma)
            computed_meta["Kernel Size"] = f"{k_size}x{k_size}"
            computed_meta["Sigma"] = f"{sigma:.1f}"

        elif operation == "Median Filtering":
            st.sidebar.caption("Non-linear order-statistic median filter (ideal for impulse noise).")
            k_size = st.sidebar.slider("Neighborhood Kernel Size (pixels)", 3, 31, 5, step=2)
            processed_img = ip.median_smoothing(orig_bgr, kernel_size=k_size)
            computed_meta["Kernel Size"] = f"{k_size}x{k_size}"

        elif operation == "Sobel Edge Detection":
            st.sidebar.caption("Calculates spatial gradient derivatives.")
            direction = st.sidebar.radio(
                "Gradient Direction:",
                ("both (Gradient Magnitude)", "x (Horizontal Gradients)", "y (Vertical Gradients)"),
            )
            ksize = st.sidebar.selectbox("Sobel Aperture Size:", [1, 3, 5, 7], index=1)
            dir_key = "both" if "both" in direction else ("x" if "x" in direction else "y")
            processed_img = ip.sobel_edge_detection(orig_bgr, ksize=ksize, direction=dir_key)
            computed_meta["Direction"] = dir_key.upper()
            computed_meta["Kernel Size"] = f"{ksize}x{ksize}"

        elif operation == "Canny Edge Detection":
            st.sidebar.caption("Multi-stage optimal edge detection with hysteresis thresholding.")
            th1 = st.sidebar.slider("Threshold 1 (Lower Hysteresis Bound)", 0, 255, 60)
            th2 = st.sidebar.slider("Threshold 2 (Upper Hysteresis Bound)", 0, 255, 160)
            processed_img = ip.canny_edge_detection(orig_bgr, threshold1=th1, threshold2=th2)
            computed_meta["Thresholds"] = f"[{min(th1, th2)}, {max(th1, th2)}]"

        elif operation == "Binary Thresholding":
            st.sidebar.caption("Segment image using a fixed global cutoff value.")
            t_val = st.sidebar.slider("Threshold Value (T)", 0, 255, 128)
            invert = st.sidebar.checkbox("Invert Binary Polarity (THRESH_BINARY_INV)", value=False)
            processed_img = ip.binary_thresholding(orig_bgr, threshold_value=t_val, inverse=invert)
            computed_meta["Threshold (T)"] = str(t_val)
            computed_meta["Inverted"] = str(invert)

        elif operation == "Otsu's Thresholding":
            st.sidebar.caption("Automatically maximizes inter-class variance between object & background.")
            invert = st.sidebar.checkbox("Invert Binary Polarity (THRESH_BINARY_INV)", value=False)
            processed_img, optimal_thresh = ip.otsu_thresholding(orig_bgr, inverse=invert)
            computed_meta["Optimal Threshold (T*)"] = f"{optimal_thresh:.1f}"
            computed_meta["Inverted"] = str(invert)

    except Exception as e:
        st.error(f"Error processing image with '{operation}': {str(e)}")
        st.stop()

    # Algorithm Explanation Card
    info = ALGORITHM_DETAILS.get(operation, {})
    with st.container():
        st.markdown(
            f"""
            <div class="info-card">
                <div style="font-size: 1.15rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.3rem;">
                    Applied Operation: {info.get('title', operation)}
                </div>
                <div style="color: #374151; font-size: 0.95rem; margin-bottom: 0.5rem;">
                    {info.get('description', '')}
                </div>
                <div style="font-size: 0.85rem; color: #4B5563;">
                    <strong>Academic Context:</strong> {info.get('syllabus_unit', 'IVP Syllabus')} &nbsp;|&nbsp;
                    <strong>Typical Use Case:</strong> {info.get('use_case', 'Image enhancement')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Main Visual Comparison Section: Side by Side
    st.subheader("Side-by-Side Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Image**")
        # OpenCV uses BGR, Streamlit st.image expects RGB for color images
        disp_orig = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB) if len(orig_bgr.shape) == 3 else orig_bgr
        st.image(disp_orig, use_container_width=True)

        dims = f"{orig_bgr.shape[1]}x{orig_bgr.shape[0]}"
        channels = "3 (Color BGR)" if len(orig_bgr.shape) == 3 else "1 (Grayscale)"
        st.markdown(
            f"""
            <div class="stat-box">
                Dimensions: <code>{dims}</code> &nbsp;|&nbsp;
                Channels: <code>{channels}</code> &nbsp;|&nbsp;
                Min / Max: <code>{np.min(orig_bgr)} / {np.max(orig_bgr)}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(f"**Processed Image ({operation})**")
        if processed_img is not None:
            disp_proc = (
                cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                if len(processed_img.shape) == 3 and processed_img.shape[2] == 3
                else processed_img
            )
            st.image(disp_proc, use_container_width=True)

            p_dims = f"{processed_img.shape[1]}x{processed_img.shape[0]}"
            p_channels = "3 (Color BGR)" if len(processed_img.shape) == 3 else "1 (Grayscale/Binary)"
            meta_str = " &nbsp;|&nbsp; ".join([f"{k}: <code>{v}</code>" for k, v in computed_meta.items()])
            extra_meta = f" &nbsp;|&nbsp; {meta_str}" if meta_str else ""
            st.markdown(
                f"""
                <div class="stat-box">
                    Dimensions: <code>{p_dims}</code> &nbsp;|&nbsp;
                    Channels: <code>{p_channels}</code> &nbsp;|&nbsp;
                    Min / Max: <code>{np.min(processed_img)} / {np.max(processed_img)}</code>
                    {extra_meta}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Download Button Section
    st.markdown("<br>", unsafe_allow_html=True)
    down_col1, down_col2, down_col3 = st.columns([1, 2, 1])
    with down_col2:
        png_bytes = ip.encode_image_to_png_bytes(processed_img)
        download_filename = f"{image_name}_{operation.lower().replace(' ', '_')}.png"
        st.download_button(
            label=f"⬇️ Download Processed Image ({download_filename})",
            data=png_bytes,
            file_name=download_filename,
            mime="image/png",
            use_container_width=True,
        )

    # Histogram Analysis Section
    st.markdown("---")
    st.subheader("Intensity Histogram Analysis")
    hist_explanation = (
        "Histograms illustrate the frequency distribution of pixel intensity levels from 0 (pure black) to 255 (pure white). "
        "Notice how operations like Contrast Stretching and Histogram Equalization modify intensity spread."
    )
    st.caption(hist_explanation)
    fig = plot_histograms(orig_bgr, processed_img)
    st.pyplot(fig)
    plt.close(fig)

    # Mathematical Formula & Theoretical Concept Expander
    with st.expander("📚 Mathematical Formulation & Algorithm Breakdown"):
        st.markdown(f"### {info.get('title', operation)}")
        st.latex(info.get("formula", ""))
        st.markdown(
            f"""
            - **Algorithm Category:** {info.get('syllabus_unit', 'Classical Digital Image Processing')}
            - **Input Representation:** 8-bit unsigned integer spatial matrix $[0, 255]$.
            - **Computational Complexity:** Linear $\\mathcal{{O}}(M \\times N)$ spatial filtering or pixelwise mapping.
            - **Relevance to NMIMS Syllabus:** Direct practical realization of university laboratory experiments and theoretical exam questions.
            """
        )

    # Syllabus Alignment Footer
    with st.expander("🎓 NMIMS B.Tech Image and Video Processing (IVP) Syllabus Alignment"):
        st.markdown(
            """
            | Module / Unit | Syllabus Topic | Implemented Tool Feature |
            | :--- | :--- | :--- |
            | **Unit 1** | Digital Image Fundamentals | Color image ingestion, matrix representation, color-to-grayscale conversion |
            | **Unit 2** | Intensity Transformations | Digital image negative, linear contrast stretching |
            | **Unit 2** | Histogram Processing | Histogram visualization, global histogram equalization (luminance-preserving) |
            | **Unit 3** | Spatial Filtering & Smoothing | 2D Gaussian convolution, non-linear median filtering for impulse noise |
            | **Unit 4** | Image Edge Detection | Sobel gradient operator (Gx, Gy, Magnitude), Canny 4-stage edge detector |
            | **Unit 5** | Image Segmentation | Global binary thresholding, Otsu's optimal inter-class variance thresholding |
            """
        )


if __name__ == "__main__":
    main()
