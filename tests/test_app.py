"""End-to-end tests of the Streamlit app.

Streamlit's AppTest cannot drive st.file_uploader, so a fake uploader serves
files registered in `uploads` under each uploader's key.
"""

import io
from pathlib import Path

import pytest
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

from tests.images import photo_like, replace_lsbs

APP = str(Path(__file__).resolve().parent.parent / "app.py")


class FakeUpload(io.BytesIO):
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name
        self.file_id = name


@pytest.fixture
def uploads(monkeypatch) -> dict[str, tuple[bytes, str]]:
    files: dict[str, tuple[bytes, str]] = {}

    def fake_uploader(label, type=None, key=None, **kwargs):
        return FakeUpload(*files[key]) if key in files else None

    monkeypatch.setattr(st, "file_uploader", fake_uploader)
    return files


def png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def run_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=120).run()


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def jpg(image: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def hide(uploads, message: str, password: str, cover: tuple[bytes, str] | None = None) -> AppTest:
    uploads["hide_upload"] = cover or (png(photo_like(0, size=200)), "holiday.png")
    at = run_app()
    at.text_area(key="hide_message").input(message).run()
    at.text_input(key="hide_password").input(password).run()
    button(at, "Hide message").click().run()
    assert not at.exception
    return at


def reveal(uploads, data: bytes, password: str, name: str = "holiday.stego.png") -> AppTest:
    uploads["reveal_upload"] = (data, name)
    at = run_app()
    at.text_input(key="reveal_password").input(password).run()
    button(at, "Reveal message").click().run()
    assert not at.exception
    return at


def test_downloaded_png_reveals_the_message(uploads):
    message = "Meet at the old bridge, 6 pm — čćž ✓"
    result = hide(uploads, message, "hunter2").session_state.hide_result

    assert result["name"] == "holiday.stego.png"
    assert [c.value for c in reveal(uploads, result["data"], "hunter2").code] == [message]


def test_wrong_password_shows_warning_not_message(uploads):
    result = hide(uploads, "secret", "right").session_state.hide_result
    at = reveal(uploads, result["data"], "wrong")

    assert not at.code
    assert "password is wrong" in at.warning[0].value


def test_hide_reports_quality_of_the_change(uploads):
    metrics = {m.label: m.value for m in hide(uploads, "hello", "pw").metric}
    assert float(metrics["PSNR"].removesuffix(" dB")) > 48
    assert float(metrics["SSIM"]) > 0.99


def test_result_disappears_when_the_message_is_edited(uploads):
    at = hide(uploads, "first", "pw")
    at.text_area(key="hide_message").input("second").run()
    assert not at.metric
    assert not [b for b in at.get("download_button")]


def test_oversized_message_disables_hiding(uploads):
    uploads["hide_upload"] = (png(Image.new("RGB", (8, 8))), "tiny.png")
    at = run_app()
    at.text_area(key="hide_message").input("x" * 100).run()
    at.text_input(key="hide_password").input("pw").run()

    assert button(at, "Hide message").disabled
    assert at.error


@pytest.mark.parametrize("name, warned", [("photo.webp", True), ("photo.png", False), ("photo.jpg", False)])
def test_lossy_uploads_are_warned_about_in_reveal(uploads, name, warned):
    uploads["reveal_upload"] = (png(photo_like(0, size=64)), name)
    assert bool(run_app().warning) is warned


def test_analyse_flags_lsb_replacement_only(uploads):
    cover = photo_like(0, size=200)
    hidden = hide(uploads, "x" * 2000, "pw").session_state.hide_result["data"]
    cases = {
        "clean": (png(cover), False),
        "replacement 50%": (png(replace_lsbs(cover, 0.5, sequential=False)), True),
        # Chi-square stays low here (~0.37), so only the RS check can flag it.
        "replacement 20%": (png(replace_lsbs(cover, 0.2, sequential=False, seed=0)), True),
        "hide tab output": (hidden, False),
    }
    for label, (data, should_flag) in cases.items():
        uploads["analyse_upload"] = (data, "test.png")
        at = run_app()
        assert not at.exception, label
        assert bool(at.error) is should_flag, label


# ---------------------------------------------------------------- JPEG


def test_jpeg_cover_stays_jpeg_and_reveals(uploads):
    cover = (jpg(photo_like(1, size=256), quality=85), "beach.jpg")
    result = hide(uploads, "Under the third palm 🌴", "sand", cover).session_state.hide_result

    assert result["name"] == "beach.stego.jpg"
    assert result["mime"] == "image/jpeg"
    assert Image.open(io.BytesIO(result["data"])).format == "JPEG"
    at = reveal(uploads, result["data"], "sand", name="beach.stego.jpg")
    assert [c.value for c in at.code] == ["Under the third palm 🌴"]


def test_analyse_flags_full_f5_but_not_clean_jpeg(uploads):
    from stego import jpeg

    clean = jpg(photo_like(0, size=384), quality=85)
    full = jpeg.embed(clean, "x" * jpeg.capacity_bytes(clean), "pw")
    for label, data, should_flag in [("clean", clean, False), ("F5 full", full, True)]:
        uploads["analyse_upload"] = (data, "photo.jpg")
        at = run_app()
        assert not at.exception, label
        assert bool(at.error) is should_flag, label
        assert "JPEG calibration" in [m.label for m in at.metric], label


def test_low_quality_jpeg_calibration_is_not_trusted(uploads):
    # Clean, but heavy compression pushes calibration to ~0.24 here.
    uploads["analyse_upload"] = (jpg(photo_like(0, size=512), quality=50), "small.jpg")
    at = run_app()
    assert not at.error
    assert any("unreliable" in c.value for c in at.caption)
