import io

import numpy as np
import pytest
from PIL import Image

from stego import lsb
from stego.crypto import OVERHEAD, DecryptionError


@pytest.fixture
def cover() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8), "RGB")


def png_roundtrip(image: Image.Image) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return Image.open(buf)


def changed_indices(cover: Image.Image, stego: Image.Image) -> np.ndarray:
    return np.flatnonzero(np.array(cover).reshape(-1) != np.array(stego).reshape(-1))


def test_message_survives_png_save_and_reload(cover):
    message = "Meet at dawn — čćšđž 🌅"
    stego = png_roundtrip(lsb.embed(cover, message, "pw"))
    assert lsb.extract(stego, "pw") == message


def test_wrong_password_finds_nothing(cover):
    stego = lsb.embed(cover, "secret", "right")
    with pytest.raises(lsb.NoMessageFound):
        lsb.extract(stego, "wrong")


def test_flipped_payload_bit_is_detected(cover):
    stego = np.array(lsb.embed(cover, "secret", "pw"))
    payload_value = lsb.positions("pw", stego.size)[lsb.HEADER.size * 8 + 40]
    stego.reshape(-1)[payload_value] ^= 1
    with pytest.raises(DecryptionError):
        lsb.extract(Image.fromarray(stego, "RGB"), "pw")


def test_capacity_is_exact_boundary(cover):
    capacity = lsb.capacity_bytes(cover)
    fits = "a" * capacity
    assert lsb.extract(lsb.embed(cover, fits, "pw"), "pw") == fits
    with pytest.raises(lsb.MessageTooLarge):
        lsb.embed(cover, fits + "a", "pw")


def test_changes_are_at_most_one_and_only_at_chosen_positions(cover):
    stego = lsb.embed(cover, "x" * 50, "pw")
    diff = np.array(stego).astype(int) - np.array(cover).astype(int)
    used_bits = (lsb.HEADER.size + 50 + OVERHEAD) * 8
    chosen = lsb.positions("pw", diff.size)[:used_bits]

    assert np.abs(diff).max() == 1
    assert set(changed_indices(cover, stego)) <= set(chosen.tolist())


def test_bits_are_spread_across_the_whole_image(cover):
    stego = lsb.embed(cover, "x" * 100, "pw")
    changed = changed_indices(cover, stego)
    total = np.array(cover).size
    # Sequential embedding would confine changes to the first ~1,500 values;
    # scattered embedding hits every quarter of the image.
    quarters = np.bincount(changed * 4 // total, minlength=4)
    assert np.all(quarters > 0)


def test_different_passwords_choose_different_positions(cover):
    a = set(changed_indices(cover, lsb.embed(cover, "x" * 100, "one")).tolist())
    b = set(changed_indices(cover, lsb.embed(cover, "x" * 100, "two")).tolist())
    assert len(a & b) < 0.2 * min(len(a), len(b))


def test_even_values_also_move_down():
    # LSB replacement can only turn an even value into value+1.
    # LSB matching must sometimes produce value-1 as well.
    cover = Image.new("RGB", (64, 64), (100, 100, 100))
    stego = np.array(lsb.embed(cover, "x" * 500, "pw"))
    assert np.any(stego == 99) and np.any(stego == 101)


@pytest.mark.parametrize("value, allowed", [(0, {0, 1}), (255, {254, 255})])
def test_extreme_values_do_not_wrap_around(value, allowed):
    cover = Image.new("RGB", (64, 64), (value, value, value))
    stego = lsb.embed(cover, "x" * 500, "pw")
    assert set(np.unique(np.array(stego)).tolist()) <= allowed
    assert lsb.extract(stego, "pw") == "x" * 500


def test_position_order_is_stable_across_versions():
    # Fixed reference values: if these change, existing stego images
    # can no longer be decoded.
    expected = [314, 838, 816, 10, 566, 97, 492, 864]
    assert lsb.positions("pw", 1000)[:8].tolist() == expected
    assert lsb.positions("pw", 1000, take=8).tolist() == expected


def test_partial_selection_matches_full_order():
    full = lsb.positions("pw", 50_000)
    for take in (1, 64, 777, 49_999):
        assert np.array_equal(lsb.positions("pw", 50_000, take=take), full[:take])


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


def test_old_sequential_format_is_reported(cover):
    flat = np.array(cover).reshape(-1)
    legacy_bits = np.unpackbits(np.frombuffer(lsb.LEGACY_MAGIC, dtype=np.uint8))
    flat[:32] = (flat[:32] & 0xFE) | legacy_bits
    legacy = Image.fromarray(flat.reshape(64, 64, 3), "RGB")
    with pytest.raises(lsb.UnsupportedFormat):
        lsb.extract(legacy, "pw")
