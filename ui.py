"""Komponen tampilan & CSS kustom (lihat docs/Design_system.md)."""
import io

import numpy as np
import streamlit as st
from PIL import Image

from processing import to_rgb_array

CSS = """
<style>
html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }
.block-container { padding-top: 2rem; max-width: 1200px; }

.hero {
    background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 60%, #06B6D4 100%);
    border-radius: 20px; padding: 2.6rem 2rem; text-align: center; color: #fff;
    box-shadow: 0 10px 30px rgba(37, 99, 235, .25); margin-bottom: 1.5rem;
}
.hero h1 { font-size: clamp(1.7rem, 8vw, 2.6rem); font-weight: 800; margin: .6rem 0 .3rem; color: #fff; overflow-wrap: anywhere; }
.hero p { font-size: 1.1rem; opacity: .92; margin: 0; }
.hero .badge {
    display: inline-block; background: rgba(255,255,255,.18); border-radius: 999px;
    padding: .25rem .9rem; font-size: .8rem; letter-spacing: .04em; margin-bottom: .4rem;
}

.card {
    background: #fff; border: 1px solid #E2E8F0; border-radius: 14px; padding: 1.1rem 1.2rem;
    box-shadow: 0 4px 16px rgba(15, 23, 42, .06); height: 100%;
}
.card h4 { margin: .2rem 0 .3rem; color: #1E3A8A; font-size: 1.05rem; }
.card p { margin: 0; color: #64748B; font-size: .92rem; }
.card .icon { font-size: 1.6rem; }

.page-header h2 { color: #1E3A8A; font-weight: 700; margin-bottom: .2rem; }
.page-header p { color: #64748B; margin-top: 0; }

.metric-card {
    background: #fff; border: 1px solid #E2E8F0; border-left: 5px solid #2563EB;
    border-radius: 12px; padding: .9rem 1.1rem; box-shadow: 0 4px 16px rgba(15,23,42,.06);
}
.metric-card .label { color: #64748B; font-size: .85rem; }
.metric-card .value { color: #0F172A; font-size: 1.6rem; font-weight: 700; }
.metric-card .sub { color: #64748B; font-size: .8rem; }

.stButton > button, .stDownloadButton > button {
    border-radius: 10px; font-weight: 600; transition: transform .1s ease, box-shadow .1s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px); box-shadow: 0 6px 16px rgba(37, 99, 235, .25);
}
[data-testid="stSidebar"] { border-right: 1px solid #E2E8F0; }
[data-testid="stImage"] img { border-radius: 10px; }
footer { visibility: hidden; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, subtitle, badge):
    st.markdown(
        f"""<div class="hero"><div class="badge">{badge}</div>
        <h1>{title}</h1><p>{subtitle}</p></div>""",
        unsafe_allow_html=True,
    )


def feature_card(icon, title, text):
    st.markdown(
        f"""<div class="card"><div class="icon">{icon}</div>
        <h4>{title}</h4><p>{text}</p></div>""",
        unsafe_allow_html=True,
    )


def page_header(title, description):
    st.markdown(
        f"""<div class="page-header"><h2>{title}</h2><p>{description}</p></div>""",
        unsafe_allow_html=True,
    )


def metric_card(label, value, sub=""):
    st.markdown(
        f"""<div class="metric-card"><div class="label">{label}</div>
        <div class="value">{value}</div><div class="sub">{sub}</div></div>""",
        unsafe_allow_html=True,
    )


def upload_image(key):
    """Pilih sumber (file / kamera) dan kembalikan array RGB uint8 (atau None)."""
    source = st.radio(
        "Sumber gambar", ["📁 Upload file", "📷 Ambil dari kamera"],
        horizontal=True, key=f"source_{key}",
    )
    if source.startswith("📁"):
        uploaded = st.file_uploader(
            "Unggah citra (JPG / PNG)", type=["jpg", "jpeg", "png"], key=f"upload_{key}"
        )
    else:
        uploaded = st.camera_input("Ambil foto", key=f"camera_{key}")
    if uploaded is None:
        st.info("Pilih atau ambil sebuah gambar untuk memulai simulasi.")
        return None
    try:
        return to_rgb_array(Image.open(uploaded))
    except Exception:
        st.error("Gambar tidak dapat dibaca. Pastikan file JPG atau PNG yang valid.")
        return None


def to_png_bytes(array):
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def show_result(original, result, caption, key):
    """Tampilkan gambar asli dan hasil berdampingan, plus tombol unduh PNG."""
    col_original, col_result = st.columns(2)
    with col_original:
        st.markdown("**Gambar Asli**")
        st.image(original, width="stretch")
    with col_result:
        st.markdown(f"**{caption}**")
        st.image(result, width="stretch", clamp=True)
        st.download_button(
            "⬇️ Unduh hasil (PNG)",
            data=to_png_bytes(result),
            file_name=f"{key}.png",
            mime="image/png",
            key=f"download_{key}",
        )
