"""Stego Lab: hide, reveal and detect messages in images.

Run with:  .venv/bin/streamlit run app.py
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from stego import lsb, metrics
from stego import steganalysis as sa
from stego.crypto import DecryptionError

IMAGE_TYPES = ["png", "bmp", "jpg", "jpeg", "webp", "tif", "tiff"]
LOSSY_TYPES = {"jpg", "jpeg", "webp"}

# Thresholds from the test-suite measurements: clean photos stay below
# chi 0.1 and RS 0.06; LSB replacement above ~10% exceeds both.
CHI_ALERT = 0.9
RS_ALERT = 0.1


def load_image(upload) -> Image.Image:
    image = Image.open(upload)
    image.load()
    # Phone photos store rotation in EXIF; apply it so the output PNG
    # (which has no EXIF) shows the same orientation.
    return ImageOps.exif_transpose(image)


def to_png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def extension(upload) -> str:
    return Path(upload.name).suffix.lstrip(".").lower()


def changes_map(cover: Image.Image, stego: Image.Image) -> Image.Image:
    """White wherever any colour value changed."""
    a = np.asarray(cover.convert("RGB"))
    b = np.asarray(stego.convert("RGB"))
    return Image.fromarray(np.any(a != b, axis=-1).astype(np.uint8) * 255, "L")


@st.cache_data(show_spinner="Running detectors…", max_entries=8)
def analyse(png_bytes: bytes) -> tuple[float, float, list[tuple[float, float]]]:
    image = Image.open(io.BytesIO(png_bytes))
    return sa.chi_square_probability(image), sa.rs_estimate(image), sa.chi_square_curve(image)


# ---------------------------------------------------------------- page

st.set_page_config(page_title="Stego Lab", page_icon="🔍", layout="wide")
st.title("Stego Lab")
st.caption(
    "Hide encrypted text in an image's least significant bits, recover it with the "
    "password, or test any image for hidden data."
)

hide_tab, reveal_tab, analyse_tab = st.tabs(["Hide", "Reveal", "Analyse"])

# ---------------------------------------------------------------- hide

with hide_tab:
    upload = st.file_uploader("Cover image", type=IMAGE_TYPES, key="hide_upload")
    if upload is not None:
        cover = load_image(upload)
        capacity = lsb.capacity_bytes(cover)

        left, right = st.columns([1, 1])
        with left:
            st.image(cover, caption=f"{cover.width} × {cover.height}", width="stretch")
        with right:
            message = st.text_area("Message", height=160, key="hide_message")
            password = st.text_input("Password", type="password", key="hide_password")
            used = len(message.encode("utf-8"))
            st.progress(min(used / capacity, 1.0) if capacity else 1.0,
                        text=f"{used:,} of {capacity:,} bytes")
            too_large = used > capacity
            if too_large:
                st.error("Message is too long for this image; use a larger image or shorten it.")

            if st.button("Hide message", type="primary",
                         disabled=not message or not password or too_large):
                with st.spinner("Encrypting and embedding…"):
                    stego = lsb.embed(cover, message, password)
                    st.session_state.hide_result = {
                        "source": (upload.file_id, message, password),
                        "png": to_png(stego),
                        "name": f"{Path(upload.name).stem}.stego.png",
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
                st.download_button("Download PNG", result["png"], file_name=result["name"],
                                   mime="image/png", type="primary", width="stretch")
                st.caption("Keep it as PNG: JPEG compression or messaging apps that "
                           "recompress images will destroy the message.")
            with st.expander("Where pixels changed"):
                st.image(result["changes"], width="stretch",
                         caption="White = at least one colour value moved by ±1. "
                                 "The password decides the positions.")

# ---------------------------------------------------------------- reveal

with reveal_tab:
    upload = st.file_uploader("Image with a hidden message", type=IMAGE_TYPES, key="reveal_upload")
    if upload is not None:
        if extension(upload) in LOSSY_TYPES:
            st.warning(f"This is a .{extension(upload)} file. Lossy compression destroys hidden "
                       "bits, so this image probably can't contain a readable message.")
        password = st.text_input("Password", type="password", key="reveal_password")
        if st.button("Reveal message", type="primary", disabled=not password):
            try:
                with st.spinner("Searching and decrypting…"):
                    secret = lsb.extract(load_image(upload), password)
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
        image = load_image(upload)
        chi_p, rs_rate, curve = analyse(to_png(image))

        flagged = chi_p > CHI_ALERT or rs_rate > RS_ALERT
        if flagged:
            st.error(f"Likely hidden data: about {rs_rate:.0%} of colour values look modified.")
        else:
            st.success("No sign of LSB replacement.")
        st.caption("Both tests detect LSB *replacement*. Images from the Hide tab use LSB "
                   "*matching*, which these tests are not designed to catch.")

        c1, c2 = st.columns(2)
        c1.metric("Chi-square", f"{chi_p:.1%}",
                  help="Probability that value pairs (2k, 2k+1) were evened out by "
                       "LSB replacement. Clean photos score near 0%.")
        c2.metric("RS estimate", f"{rs_rate:.1%}",
                  help="Estimated share of colour values carrying hidden bits. "
                       "Accurate to about ±8 points up to 70%.")

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
