"""Fitur kamera real-time (opsional).

Modul ini sengaja terpisah dan aman di-import tanpa `streamlit-webrtc`/`av`:
paket tersebut hanya dimuat saat halaman Live Camera dibuka. Jika tidak
terpasang atau gagal dimuat, aplikasi utama tetap berjalan normal.

Install dependensi opsional:  pip install -r requirements-live.txt
"""
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

import processing as proc

CASCADE_PATH = Path(__file__).parent / "haarcascade_frontalface_default.xml"
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

EFFECTS = [
    "Normal", "Grayscale", "Negative", "Sepia", "Emboss",
    "Edge Detection (Canny)", "Blur (Gaussian)", "Face Detection",
]

_cascade = None


def _get_cascade():
    global _cascade
    if _cascade is None:
        cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
        if cascade.empty():
            raise FileNotFoundError(f"Haar cascade tidak ditemukan: {CASCADE_PATH}")
        _cascade = cascade
    return _cascade


def apply_effect(rgb, effect):
    """Terapkan efek pada satu frame RGB uint8; kembalikan RGB uint8 (H x W x 3)."""
    if effect == "Normal":
        return rgb
    if effect == "Grayscale":
        return proc.grayscale(rgb)
    if effect == "Negative":
        return proc.negative(rgb)
    if effect == "Sepia":
        return proc.sepia(rgb)
    if effect == "Emboss":
        return cv2.cvtColor(proc.emboss(rgb), cv2.COLOR_GRAY2RGB)
    if effect == "Edge Detection (Canny)":
        return cv2.cvtColor(proc.edge_detection(rgb, "Canny"), cv2.COLOR_GRAY2RGB)
    if effect == "Blur (Gaussian)":
        return proc.image_filtering(rgb, "Gaussian Filter", 15)
    if effect == "Face Detection":
        output = rgb.copy()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        for x, y, w, h in _get_cascade().detectMultiScale(gray, 1.1, 5, minSize=(40, 40)):
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
        return output
    raise ValueError(f"Efek tidak dikenal: {effect}")


def _load_webrtc():
    """Muat dependensi opsional. Kembalikan (webrtc_streamer, av, VideoProcessorBase)."""
    import av
    from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

    return webrtc_streamer, av, VideoProcessorBase


def render_live(img=None, key="live"):
    """Halaman Live Camera. Parameter `img` diabaikan (menyamakan tanda tangan renderer)."""
    try:
        webrtc_streamer, av, VideoProcessorBase = _load_webrtc()
    except Exception as error:
        st.warning(
            "Fitur Live Camera membutuhkan paket opsional yang belum terpasang "
            "atau gagal dimuat."
        )
        st.code("pip install -r requirements-live.txt", language="bash")
        st.caption(f"Detail: {type(error).__name__}: {error}")
        st.info("Fitur lain, termasuk **Ambil dari kamera** (foto), tetap dapat dipakai.")
        return

    class EffectProcessor(VideoProcessorBase):
        def __init__(self):
            self.effect = "Normal"

        def recv(self, frame):
            rgb = frame.to_ndarray(format="rgb24")
            try:
                out = apply_effect(rgb, self.effect)
            except Exception:  # satu frame gagal tidak boleh menghentikan stream
                out = rgb
            return av.VideoFrame.from_ndarray(np.ascontiguousarray(out), format="rgb24")

    effect = st.selectbox("Efek", EFFECTS, key=f"{key}_effect")
    ctx = webrtc_streamer(
        key="live-camera",
        video_processor_factory=EffectProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    if ctx.video_processor:
        ctx.video_processor.effect = effect
    st.caption(
        "Klik **START** dan izinkan akses kamera. Butuh HTTPS atau localhost, "
        "dan koneksi ke server STUN."
    )
