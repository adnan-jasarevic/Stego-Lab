"""Stego Lab: hide, reveal and detect messages in images.

Run with:  .venv/bin/streamlit run app.py
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from stego import jpeg, lsb, metrics
from stego import steganalysis as sa
from stego.crypto import DecryptionError

IMAGE_TYPES = ["png", "bmp", "jpg", "jpeg", "webp", "tif", "tiff"]
LOSSY_TYPES = {"webp"}

# Thresholds from the test-suite measurements: clean photos stay below
# chi 0.1 and RS 0.06; LSB replacement above ~10% exceeds both.
CHI_ALERT = 0.9
RS_ALERT = 0.1
# Clean JPEGs (quality >= 70) stay below 0.06; F5 at >= 40% of capacity
# scores above 0.15. Below quality 70 clean images reach 0.25.
CALIBRATION_ALERT = 0.1
CALIBRATION_MIN_QUALITY = 70


def load_image(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    image.load()
    # Phone photos store rotation in EXIF; apply it so previews, metrics
    # and the output PNG (which has no EXIF) show the same orientation.
    return ImageOps.exif_transpose(image)


def to_png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def changes_map(cover: Image.Image, stego: Image.Image) -> Image.Image:
    """White wherever any colour value changed."""
    a = np.asarray(cover.convert("RGB"))
    b = np.asarray(stego.convert("RGB"))
    return Image.fromarray(np.any(a != b, axis=-1).astype(np.uint8) * 255, "L")


@st.cache_data(show_spinner=False, max_entries=8)
def jpeg_capacity(data: bytes) -> int:
    return jpeg.capacity_bytes(data)


@st.cache_data(show_spinner="Running detectors…", max_entries=8)
def analyse(png_bytes: bytes) -> tuple[float, float, list[tuple[float, float]]]:
    image = Image.open(io.BytesIO(png_bytes))
    return sa.chi_square_probability(image), sa.rs_estimate(image), sa.chi_square_curve(image)


@st.cache_data(show_spinner="Calibrating JPEG…", max_entries=8)
def analyse_jpeg(data: bytes) -> tuple[float, int]:
    return sa.jpeg_calibration_estimate(data), jpeg.estimate_quality(data)


# ---------------------------------------------------------------- page

st.set_page_config(page_title="Stego Lab", page_icon="🔍", layout="wide")
st.title("Stego Lab")
st.caption(
    "Hide encrypted text in PNG pixels or JPEG DCT coefficients, recover it with the "
    "password, or test any image for hidden data."
)

hide_tab, reveal_tab, analyse_tab = st.tabs(["Hide", "Reveal", "Analyse"])

# ---------------------------------------------------------------- hide

with hide_tab:
    upload = st.file_uploader("Cover image", type=IMAGE_TYPES, key="hide_upload")
    if upload is not None:
        data = upload.getvalue()
        cover = load_image(data)
        is_jpeg = jpeg.is_jpeg(data)
        capacity = jpeg_capacity(data) if is_jpeg else lsb.capacity_bytes(cover)
        method = ("JPEG · hidden in DCT coefficients (F5), stays a JPEG" if is_jpeg
                  else "hidden in pixel LSBs, saved as PNG")

        left, right = st.columns([1, 1])
        with left:
            st.image(cover, caption=f"{cover.width} × {cover.height} · {method}", width="stretch")
        with right:
            message = st.text_area("Message", height=160, key="hide_message")
            password = st.text_input("Password", type="password", key="hide_password")
            used = len(message.encode("utf-8"))
            st.progress(min(used / capacity, 1.0) if capacity else 1.0,
                        text=f"{used:,} of {'about ' if is_jpeg else ''}{capacity:,} bytes")
            too_large = used > capacity
            if too_large:
                st.error("Message is too long for this image; use a larger image or shorten it.")

            if st.button("Hide message", type="primary",
                         disabled=not message or not password or too_large):
                try:
                    with st.spinner("Encrypting and embedding…"):
                        if is_jpeg:
                            output = jpeg.embed(data, message, password)
                            stego = load_image(output)
                            suffix, mime = "jpg", "image/jpeg"
                        else:
                            stego = lsb.embed(cover, message, password)
                            output = to_png(stego)
                            suffix, mime = "png", "image/png"
                except lsb.MessageTooLarge as exc:
                    st.error(str(exc))
                else:
                    st.session_state.hide_result = {
                        "source": (upload.file_id, message, password),
                        "data": output,
                        "name": f"{Path(upload.name).stem}.stego.{suffix}",
                        "mime": mime,
                        "mse": metrics.mse(cover, stego),
                        "psnr": metrics.psnr(cover, stego),
                        "ssim": metrics.ssim(cover, stego),
                        "changes": to_png(changes_map(cover, stego)),
                    }

        result = st.session_state.get("hide_result")
        if result and result["source"] == (upload.file_id, message, password):
            st.divider()
            m1, m2, m3, dl = st.columns(4)
            m1.metric("MSE", f"{result['mse']:.4f}", help="Mean squared error per colour value. 0 = identical.")
            m2.metric("PSNR", f"{result['psnr']:.2f} dB", help="Above ~40 dB the change is invisible.")
            m3.metric("SSIM", f"{result['ssim']:.6f}", help="Structural similarity. 1 = identical.")
            with dl:
                kind = "JPEG" if result["mime"] == "image/jpeg" else "PNG"
                st.download_button(f"Download {kind}", result["data"], file_name=result["name"],
                                   mime=result["mime"], type="primary", width="stretch")
                if kind == "JPEG":
                    st.caption("Original quality and metadata are kept. Editing, resizing or "
                               "re-saving it, including by apps that recompress photos, "
                               "destroys the message.")
                else:
                    st.caption("Keep it as PNG: JPEG compression or messaging apps that "
                               "recompress images will destroy the message.")
            with st.expander("Where pixels changed"):
                st.image(result["changes"], width="stretch",
                         caption="White = at least one colour value changed. JPEG changes "
                                 "affect whole 8 × 8 blocks. The password decides the positions.")

# ---------------------------------------------------------------- reveal

with reveal_tab:
    upload = st.file_uploader("Image with a hidden message", type=IMAGE_TYPES, key="reveal_upload")
    if upload is not None:
        suffix = Path(upload.name).suffix.lstrip(".").lower()
        if suffix in LOSSY_TYPES:
            st.warning(f"This is a .{suffix} file. Lossy compression destroys hidden bits, "
                       "so this image probably can't contain a readable message.")
        password = st.text_input("Password", type="password", key="reveal_password")
        if st.button("Reveal message", type="primary", disabled=not password):
            data = upload.getvalue()
            try:
                with st.spinner("Searching and decrypting…"):
                    if jpeg.is_jpeg(data):
                        secret = jpeg.extract(data, password)
                    else:
                        secret = lsb.extract(load_image(data), password)
            except lsb.NoMessageFound as exc:
                st.warning(str(exc))
            except lsb.UnsupportedFormat as exc:
                st.error(str(exc))
            except DecryptionError:
                st.error("A message was found, but it has been damaged or altered and "
                         "can't be decrypted.")
            else:
                st.code(secret, language=None, wrap_lines=True)

# ---------------------------------------------------------------- analyse

with analyse_tab:
    upload = st.file_uploader("Image to test", type=IMAGE_TYPES, key="analyse_upload")
    if upload is not None:
        data = upload.getvalue()
        image = load_image(data)
        chi_p, rs_rate, curve = analyse(to_png(image))
        calibration, quality = analyse_jpeg(data) if jpeg.is_jpeg(data) else (None, None)
        calibration_reliable = quality is not None and quality >= CALIBRATION_MIN_QUALITY

        findings = []
        if chi_p > CHI_ALERT or rs_rate > RS_ALERT:
            findings.append(f"about {rs_rate:.0%} of colour values look modified (LSB replacement)")
        if calibration_reliable and calibration > CALIBRATION_ALERT:
            findings.append(f"about {calibration:.0%} of DCT coefficients look modified "
                            "(F5-style JPEG embedding)")
        if findings:
            st.error("Likely hidden data: " + "; ".join(findings) + ".")
        else:
            st.success("No sign of hidden data.")

        notes = ["Chi-square and RS detect LSB *replacement*; PNGs from the Hide tab use LSB "
                 "*matching*, which they are not designed to catch."]
        if quality is not None:
            notes.append("Calibration catches F5 when a JPEG is filled to roughly 40% of its "
                         "capacity or more; smaller messages are hard to detect.")
            if not calibration_reliable:
                notes.append(f"This JPEG is heavily compressed (quality ≈ {quality}), so "
                             "calibration is unreliable and isn't used for the verdict.")
        st.caption(" ".join(notes))

        columns = st.columns(3 if quality is not None else 2)
        columns[0].metric("Chi-square", f"{chi_p:.1%}",
                          help="Probability that value pairs (2k, 2k+1) were evened out by "
                               "LSB replacement. Clean photos score near 0%.")
        columns[1].metric("RS estimate", f"{rs_rate:.1%}",
                          help="Estimated share of colour values carrying hidden bits. "
                               "Accurate to about ±8 points up to 70%.")
        if quality is not None:
            columns[2].metric("JPEG calibration", f"{calibration:.1%}",
                              help=f"Estimated share of DCT coefficients changed by F5-style "
                                   f"embedding. JPEG quality ≈ {quality}. Accurate to about "
                                   "±10 points at quality 70 and above.")

        st.subheader("Chi-square along the image")
        st.caption("Scanned row by row from the top. A message written from the start "
                   "keeps this near 100% until the message ends, then it drops.")
        st.line_chart(
            pd.DataFrame(
                {"Probability (%)": [p * 100 for _, p in curve]},
                index=pd.Index([f * 100 for f, _ in curve], name="Image scanned (%)"),
            ),
            x_label="Image scanned (%)",
            y_label="Probability (%)",
        )

        st.subheader("Bit planes")
        ch_col, bit_col = st.columns([1, 3])
        channel = ch_col.segmented_control("Channel", list(sa.CHANNELS), default="R",
                                           key="plane_channel") or "R"
        bit = bit_col.slider("Bit (0 = least significant)", 0, 7, 0, key="plane_bit")
        left, right = st.columns(2)
        left.image(image, width="stretch")
        right.image(sa.bit_plane(image, channel, bit), width="stretch",
                    caption="Natural images show shapes here; hidden data looks like static.")
