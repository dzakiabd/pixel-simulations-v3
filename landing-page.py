import streamlit as st

import ui
from mulai import show as show_mulai

FEATURES = [
    ("🎨", "Pre-processing", "Grayscale, histogram, equalization, filtering, edge detection, dan sharpening."),
    ("🧩", "Segmentasi & Fitur", "Thresholding, Otsu, color moment, HOG, dan Local Binary Pattern."),
    ("✨", "Efek Visual", "Emboss dan sepia untuk eksperimen cepat."),
]


def _open_app(session_state):
    session_state["page"] = "mulai"


def home(session_state):
    ui.inject_css()
    ui.hero(
        "PixelSimulations",
        "Simulasi interaktif pengolahan citra digital — unggah gambar, atur parameter, lihat hasilnya.",
        "ITENAS × KAMPUS MERDEKA",
    )

    for column, (icon, title, text) in zip(st.columns(len(FEATURES)), FEATURES):
        with column:
            ui.feature_card(icon, title, text)

    st.write("")
    _, center, _ = st.columns([1, 1, 1])
    with center:
        st.button("🚀 Mulai Simulasi", type="primary", width="stretch",
                  on_click=_open_app, args=(session_state,), key="mulai_btn")

    st.write("")
    _, logo_left, logo_right, _ = st.columns([2, 1, 1, 2], vertical_alignment="center")
    logo_left.image("img/itenas.png", width=130)
    logo_right.image("img/kampus-merdeka.png", width=130)


def main():
    st.set_page_config(page_title="PixelSimulations", page_icon="🖼️", layout="wide")
    session_state = st.session_state
    if session_state.get("page") == "mulai":
        show_mulai(session_state)
    else:
        home(session_state)


if __name__ == "__main__":
    main()
