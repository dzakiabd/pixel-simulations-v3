"""Fungsi pemrosesan citra murni (tanpa Streamlit).

Konvensi: semua fungsi menerima array numpy RGB uint8 (H x W x 3) dan
mengembalikan array uint8 (RGB atau grayscale 2D) kecuali dinyatakan lain.
"""
import cv2
import numpy as np
from PIL import Image
from skimage import exposure
from skimage.feature import hog, local_binary_pattern

MAX_OUTPUT_SIDE = 4000
MAX_HOG_SIDE = 512


def to_rgb_array(image):
    """Normalisasi gambar PIL ke array RGB uint8 (menangani RGBA, P, L, dsb)."""
    if isinstance(image, Image.Image):
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba)
        return np.array(image.convert("RGB"), dtype=np.uint8)

    array = np.asarray(image)
    if array.ndim == 2:
        return cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if array.shape[2] == 4:
        return array[:, :, :3].astype(np.uint8)
    return array.astype(np.uint8)


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def _gray_to_rgb(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def _to_uint8(values):
    return np.clip(values, 0, 255).astype(np.uint8)


def _normalize_to_uint8(values):
    """Skalakan array float ke 0-255; array konstan menjadi hitam."""
    values = np.asarray(values, dtype=np.float64)
    low, high = values.min(), values.max()
    if high - low < 1e-12:
        return np.zeros(values.shape, dtype=np.uint8)
    return ((values - low) / (high - low) * 255).astype(np.uint8)


def _odd_kernel(size):
    size = max(1, int(size))
    return size if size % 2 == 1 else size + 1


# --------------------------------------------------------------------------
# Pre-processing
# --------------------------------------------------------------------------
def grayscale(img):
    return _gray_to_rgb(_gray(img))


def gray_histogram(img):
    gray = _gray(img)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    return gray, hist.flatten()


def rgb_histograms(img):
    return tuple(
        cv2.calcHist([img], [channel], None, [256], [0, 256]).flatten()
        for channel in range(3)
    )


def histogram_equalization(img):
    """Equalisasi pada kanal luminans (Y) agar warna tidak bergeser."""
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def contrast_stretching(img):
    data = img.astype(np.float64)
    low, high = data.min(), data.max()
    if high == low:
        return img.copy()
    return _to_uint8((data - low) / (high - low) * 255)


def brightness_adjustment(img, value):
    """Tambahkan `value` (-255..255) ke setiap piksel."""
    return _to_uint8(img.astype(np.int16) + int(value))


def resize_with_methods(img, scale):
    """Ubah ukuran dengan tiga metode interpolasi. `scale` berupa rasio (mis. 0.5)."""
    height, width = img.shape[:2]
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    longest = max(new_width, new_height)
    if longest > MAX_OUTPUT_SIDE:
        ratio = MAX_OUTPUT_SIDE / longest
        new_width = max(1, int(new_width * ratio))
        new_height = max(1, int(new_height * ratio))

    pil_image = Image.fromarray(img)
    methods = {
        "Nearest Neighbor": Image.Resampling.NEAREST,
        "Bilinear": Image.Resampling.BILINEAR,
        "Bicubic": Image.Resampling.BICUBIC,
    }
    return {
        name: np.array(pil_image.resize((new_width, new_height), resample=method))
        for name, method in methods.items()
    }


def black_and_white(img, threshold=128):
    _, binary = cv2.threshold(_gray(img), threshold, 255, cv2.THRESH_BINARY)
    return _gray_to_rgb(binary)


def negative(img):
    return 255 - img


FILTER_TYPES = ["Mean Filter", "Gaussian Filter", "Median Filter", "Max Filter", "Min Filter"]


def image_filtering(img, filter_type, kernel_size):
    size = _odd_kernel(kernel_size)
    if filter_type == "Mean Filter":
        return cv2.blur(img, (size, size))
    if filter_type == "Gaussian Filter":
        return cv2.GaussianBlur(img, (size, size), 0)
    if filter_type == "Median Filter":
        return cv2.medianBlur(img, size)
    kernel = np.ones((size, size), np.uint8)
    if filter_type == "Max Filter":
        return cv2.dilate(img, kernel)
    if filter_type == "Min Filter":
        return cv2.erode(img, kernel)
    raise ValueError(f"Filter tidak dikenal: {filter_type}")


EDGE_TYPES = ["Canny", "Sobel", "Prewitt", "Scharr", "Laplacian"]


def edge_detection(img, filter_type):
    gray = _gray(img)
    if filter_type == "Canny":
        edges = cv2.Canny(gray, 100, 200)
    elif filter_type == "Sobel":
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = _normalize_to_uint8(np.hypot(gx, gy))
    elif filter_type == "Prewitt":
        kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float64)
        ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float64)
        gx = cv2.filter2D(gray, cv2.CV_64F, kx)
        gy = cv2.filter2D(gray, cv2.CV_64F, ky)
        edges = _normalize_to_uint8(np.hypot(gx, gy))
    elif filter_type == "Scharr":
        gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        edges = _normalize_to_uint8(np.hypot(gx, gy))
    elif filter_type == "Laplacian":
        edges = _normalize_to_uint8(np.abs(cv2.Laplacian(gray, cv2.CV_64F)))
    else:
        raise ValueError(f"Filter tidak dikenal: {filter_type}")
    return edges


SHARPEN_TYPES = ["Unsharp Mask", "Laplacian Sharpening", "Kernel Sharpen"]


def image_sharpening(img, filter_type, amount=1.0):
    """Pertajam citra. `amount` mengatur kekuatan efek (0 = tanpa perubahan)."""
    data = img.astype(np.float64)
    if filter_type == "Unsharp Mask":
        blurred = cv2.GaussianBlur(data, (0, 0), 2.0)
        sharpened = data + amount * (data - blurred)
    elif filter_type == "Laplacian Sharpening":
        laplacian = cv2.Laplacian(data, cv2.CV_64F, ksize=3)
        sharpened = data - amount * laplacian
    elif filter_type == "Kernel Sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)
        sharpened = data + amount * (cv2.filter2D(data, cv2.CV_64F, kernel) - data)
    else:
        raise ValueError(f"Filter tidak dikenal: {filter_type}")
    return _to_uint8(sharpened)


# --------------------------------------------------------------------------
# Segmentasi & ekstraksi fitur
# --------------------------------------------------------------------------
def global_thresholding(img, value):
    _, binary = cv2.threshold(_gray(img), int(value), 255, cv2.THRESH_BINARY)
    return _gray_to_rgb(binary)


def otsu_threshold(img):
    threshold, binary = cv2.threshold(_gray(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _gray_to_rgb(binary), float(threshold)


def color_moments(img):
    """Mean dan standar deviasi kanal H (derajat 0-360), S, V (0-255)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float64)
    hsv[:, :, 0] *= 2.0  # OpenCV menyimpan hue 0-179
    names = ("Hue", "Saturation", "Value")
    return {
        name: (float(hsv[:, :, i].mean()), float(hsv[:, :, i].std()))
        for i, name in enumerate(names)
    }


def hog_image(img):
    gray = _gray(img)
    longest = max(gray.shape)
    if longest > MAX_HOG_SIDE:
        ratio = MAX_HOG_SIDE / longest
        gray = cv2.resize(gray, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    _, hog_vis = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2), visualize=True)
    rescaled = exposure.rescale_intensity(hog_vis, out_range=(0, 1))
    return (rescaled * 255).astype(np.uint8)


def local_binary_pattern_image(img, radius=3):
    gray = _gray(img)
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    return (lbp / (n_points + 1) * 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Efek lain
# --------------------------------------------------------------------------
def emboss(img):
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    embossed = cv2.filter2D(_gray(img).astype(np.float32), -1, kernel) + 128
    return _to_uint8(embossed)


def sepia(img):
    # Baris = keluaran R, G, B; kolom = masukan R, G, B
    matrix = np.array(
        [[0.393, 0.769, 0.189],
         [0.349, 0.686, 0.168],
         [0.272, 0.534, 0.131]],
        dtype=np.float64,
    )
    return _to_uint8(img.astype(np.float64) @ matrix.T)
