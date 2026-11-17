import io

import numpy as np
import pytest
from PIL import Image

from stego import lsb
from stego.crypto import DecryptionError


@pytest.fixture
def cover() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8), "RGB")


def png_roundtrip(image: Image.Image) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return Image.open(buf)


def test_message_survives_png_save_and_reload(cover):
    message = "Meet at dawn — čćšđž 🌅"
    stego = png_roundtrip(lsb.embed(cover, message, "pw"))
    assert lsb.extract(stego, "pw") == message


def test_wrong_password_is_rejected(cover):
    stego = lsb.embed(cover, "secret", "right")
    with pytest.raises(DecryptionError):
        lsb.extract(stego, "wrong")


def test_flipped_payload_bit_is_detected(cover):
    stego = np.array(lsb.embed(cover, "secret", "pw"))
    payload_bit = lsb.HEADER.size * 8 + 40
    stego.reshape(-1)[payload_bit] ^= 1
    with pytest.raises(DecryptionError):
        lsb.extract(Image.fromarray(stego, "RGB"), "pw")


def test_capacity_is_exact_boundary(cover):
    capacity = lsb.capacity_bytes(cover)
    fits = "a" * capacity
    assert lsb.extract(lsb.embed(cover, fits, "pw"), "pw") == fits
    with pytest.raises(lsb.MessageTooLarge):
        lsb.embed(cover, fits + "a", "pw")


def test_only_lowest_bit_of_used_values_changes(cover):
    original = np.array(cover).reshape(-1).astype(int)
    stego = np.array(lsb.embed(cover, "x" * 50, "pw")).reshape(-1).astype(int)
    used_bits = (lsb.HEADER.size + 50 + lsb.crypto.OVERHEAD) * 8

    assert np.all(np.abs(stego[:used_bits] - original[:used_bits]) <= 1)
    assert np.array_equal(stego[used_bits:], original[used_bits:])


def test_alpha_channel_is_preserved():
    rng = np.random.default_rng(1)
    rgba = Image.fromarray(rng.integers(0, 256, (32, 32, 4), dtype=np.uint8), "RGBA")
    stego = lsb.embed(rgba, "hi", "pw")

    assert stego.mode == "RGBA"
    assert np.array_equal(np.array(stego)[..., 3], np.array(rgba)[..., 3])
    assert lsb.extract(stego, "pw") == "hi"


def test_clean_image_reports_no_message(cover):
    with pytest.raises(lsb.NoMessageFound):
        lsb.extract(cover, "pw")
