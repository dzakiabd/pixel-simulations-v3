import matplotlib.pyplot as plt
import streamlit as st

import processing as proc
import ui


def _go_home(session_state):
    session_state["page"] = None


def _plot_histograms(histograms, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    for hist, color, label in zip(histograms, ("red", "green", "blue"), "RGB"):
        ax.plot(hist, color=color, label=label)
    ax.set_title(title)
    ax.set_xlabel("Intensitas Piksel")
    ax.set_ylabel("Jumlah Piksel")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------
# Renderer per proses: menerima `img` (RGB uint8) dan `key` (id unik)
# --------------------------------------------------------------------------
def render_grayscale(img, key):
    ui.show_result(img, proc.grayscale(img), "Hasil Grayscale", key)


def render_histogram(img, key):
    gray, hist = proc.gray_histogram(img)
    ui.show_result(img, gray, "Citra Grayscale", key)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist, color="#2563EB")
    ax.fill_between(range(256), hist, color="#2563EB", alpha=0.15)
    ax.set_title("Histogram Grayscale")
    ax.set_xlabel("Intensitas Piksel")
    ax.set_ylabel("Jumlah Piksel")
    st.pyplot(fig)
    plt.close(fig)


def render_equalization(img, key):
    result = proc.histogram_equalization(img)
    ui.show_result(img, result, "Hasil Histogram Equalization", key)
    col_before, col_after = st.columns(2)
    with col_before:
        _plot_histograms(proc.rgb_histograms(img), "Histogram RGB Sebelum")
    with col_after:
        _plot_histograms(proc.rgb_histograms(result), "Histogram RGB Sesudah")


def render_contrast(img, key):
    ui.show_result(img, proc.contrast_stretching(img), "Hasil Contrast Stretching", key)


def render_brightness(img, key):
    value = st.slider("Brightness", -255, 255, 0, 1, key=f"{key}_value")
    ui.show_result(img, proc.brightness_adjustment(img, value),
                   f"Brightness {value:+d}", key)


def render_interpolation(img, key):
    scale_percent = st.slider("Skala ukuran (%)", 10, 400, 50, 10, key=f"{key}_scale")
    results = proc.resize_with_methods(img, scale_percent / 100)
    height, width = next(iter(results.values())).shape[:2]
    st.caption(f"Ukuran asli {img.shape[1]}×{img.shape[0]} → hasil {width}×{height} piksel. "
               "Perbedaan terlihat jelas saat diperbesar.")
    columns = st.columns(len(results))
    for column, (name, result) in zip(columns, results.items()):
        with column:
            st.markdown(f"**{name}**")
            st.image(result, width="stretch")
            st.download_button("⬇️ Unduh", data=ui.to_png_bytes(result),
                               file_name=f"{key}_{name.split()[0].lower()}.png",
                               mime="image/png", key=f"{key}_dl_{name}")


def render_black_white(img, key):
    threshold = st.slider("Ambang batas", 0, 255, 128, 1, key=f"{key}_threshold")
    ui.show_result(img, proc.black_and_white(img, threshold), "Hasil Black and White", key)


def render_negative(img, key):
    ui.show_result(img, proc.negative(img), "Hasil Negative", key)


def render_filtering(img, key):
    col_a, col_b = st.columns(2)
    with col_a:
        filter_type = st.selectbox("Tipe filter", proc.FILTER_TYPES, key=f"{key}_type")
    with col_b:
        kernel_size = st.slider("Ukuran kernel", 3, 15, 3, 2, key=f"{key}_kernel")
    ui.show_result(img, proc.image_filtering(img, filter_type, kernel_size),
                   f"Hasil {filter_type}", key)


def render_edge(img, key):
    filter_type = st.selectbox("Tipe filter", proc.EDGE_TYPES, key=f"{key}_type")
    ui.show_result(img, proc.edge_detection(img, filter_type),
                   f"Hasil {filter_type} Edge Detection", key)


def render_sharpening(img, key):
    col_a, col_b = st.columns(2)
    with col_a:
        filter_type = st.selectbox("Metode", proc.SHARPEN_TYPES, key=f"{key}_type")
    with col_b:
        amount = st.slider("Kekuatan", 0.0, 3.0, 1.0, 0.1, key=f"{key}_amount")
    ui.show_result(img, proc.image_sharpening(img, filter_type, amount),
                   f"Hasil {filter_type}", key)


def render_global_threshold(img, key):
    value = st.slider("Threshold", 0, 255, 128, 1, key=f"{key}_value")
    ui.show_result(img, proc.global_thresholding(img, value),
                   f"Global Threshold = {value}", key)


def render_otsu(img, key):
    result, threshold = proc.otsu_threshold(img)
    ui.show_result(img, result, f"Otsu Thresholding (T = {threshold:.0f})", key)


def render_color_moment(img, key):
    st.image(img, caption="Gambar Asli", width="stretch")
    moments = proc.color_moments(img)
    units = {"Hue": "derajat (0–360)", "Saturation": "skala 0–255", "Value": "skala 0–255"}
    columns = st.columns(3)
    for column, (name, (mean, std)) in zip(columns, moments.items()):
        with column:
            ui.metric_card(f"{name} Mean", f"{mean:.2f}", f"Std {std:.2f} · {units[name]}")


def render_color_histogram(img, key):
    st.image(img, caption="Gambar Asli", width="stretch")
    _plot_histograms(proc.rgb_histograms(img), "Histogram RGB")


def render_hog(img, key):
    ui.show_result(img, proc.hog_image(img), "Histogram of Oriented Gradients", key)


def render_lbp(img, key):
    ui.show_result(img, proc.local_binary_pattern_image(img), "Local Binary Pattern", key)


def render_emboss(img, key):
    ui.show_result(img, proc.emboss(img), "Hasil Emboss", key)


def render_sepia(img, key):
    ui.show_result(img, proc.sepia(img), "Hasil Sepia", key)


# kategori -> {nama proses: (deskripsi, renderer)}
MENU = {
    "Pre-processing": {
        "Grayscale": ("Mengubah citra berwarna menjadi skala keabuan.", render_grayscale),
        "Histogram": ("Distribusi intensitas piksel citra grayscale.", render_histogram),
        "Histogram Equalization": ("Meratakan distribusi intensitas untuk menaikkan kontras.", render_equalization),
        "Contrast Stretching": ("Merentangkan intensitas ke rentang penuh 0–255.", render_contrast),
        "Brightness Adjustment": ("Menambah atau mengurangi kecerahan citra.", render_brightness),
        "Image Interpolation": ("Membandingkan metode interpolasi saat mengubah ukuran.", render_interpolation),
        "Black and White": ("Binerisasi citra dengan ambang batas tertentu.", render_black_white),
        "Negative Image": ("Membalik nilai intensitas setiap piksel.", render_negative),
        "Image Filtering": ("Filter spasial: mean, gaussian, median, max, min.", render_filtering),
        "Edge Detection": ("Mendeteksi tepi objek dengan berbagai operator.", render_edge),
        "Image Sharpening": ("Mempertajam detail citra.", render_sharpening),
    },
    "Image Segmentation & Feature Extraction": {
        "Global Thresholding": ("Segmentasi dengan satu nilai ambang tetap.", render_global_threshold),
        "Otsu Thresholding": ("Ambang batas optimal dihitung otomatis.", render_otsu),
        "Color Moment": ("Statistik warna (mean & std) pada ruang HSV.", render_color_moment),
        "Color Histogram": ("Histogram per kanal warna R, G, B.", render_color_histogram),
        "Histogram Orientation Gradient": ("Fitur HOG berbasis arah gradien.", render_hog),
        "Local Binary Pattern": ("Fitur tekstur LBP (uniform).", render_lbp),
    },
    "Other": {
        "Emboss": ("Efek timbul pada citra.", render_emboss),
        "Sepia": ("Efek warna sepia klasik.", render_sepia),
    },
}


def show(session_state):
    ui.inject_css()

    with st.sidebar:
        col1, col2 = st.columns(2)
        col1.image("img/itenas-logo.png", width=90)
        col2.image("img/kampus-merdeka.png", width=90)
        st.markdown("### PixelSimulations")
        category = st.selectbox("Kategori", list(MENU.keys()))
        choice = st.selectbox("Proses", list(MENU[category].keys()))
        st.button("← Kembali ke beranda", on_click=_go_home, args=(session_state,),
                  width="stretch")

    description, renderer = MENU[category][choice]
    ui.page_header(choice, description)

    img = ui.upload_image(choice)
    if img is not None:
        key = choice.lower().replace(" ", "_")
        try:
            renderer(img, key)
        except Exception as error:  # tampilkan pesan ramah, bukan stack trace
            st.error(f"Terjadi kesalahan saat memproses gambar: {error}")
