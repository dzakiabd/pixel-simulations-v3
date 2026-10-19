import numpy as np
import pytest

import live_camera


@pytest.fixture
def frame():
    return np.random.default_rng(0).integers(0, 256, (120, 160, 3), dtype=np.uint8)


@pytest.mark.parametrize("effect", live_camera.EFFECTS)
def test_apply_effect_keeps_frame_shape(frame, effect):
    out = live_camera.apply_effect(frame, effect)
    assert out.shape == frame.shape and out.dtype == np.uint8


def test_normal_is_identity(frame):
    assert np.array_equal(live_camera.apply_effect(frame, "Normal"), frame)


def test_face_detection_does_not_mutate_input(frame):
    before = frame.copy()
    live_camera.apply_effect(frame, "Face Detection")
    assert np.array_equal(frame, before)


def test_unknown_effect_raises(frame):
    with pytest.raises(ValueError):
        live_camera.apply_effect(frame, "Nope")


def test_cascade_file_exists():
    assert live_camera.CASCADE_PATH.exists()
    assert not live_camera._get_cascade().empty()


def test_render_degrades_gracefully_without_webrtc(monkeypatch):
    """Tanpa streamlit-webrtc/av, render_live tidak boleh melempar exception."""
    def boom():
        raise ModuleNotFoundError("streamlit_webrtc")
    monkeypatch.setattr(live_camera, "_load_webrtc", boom)
    shown = []
    for name in ("warning", "code", "caption", "info"):
        monkeypatch.setattr(live_camera.st, name, lambda *a, **k: shown.append(a), raising=True)
    live_camera.render_live()
    assert shown
