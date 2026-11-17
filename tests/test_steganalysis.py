import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import gaussian_filter

from stego import lsb
from stego import steganalysis as sa


def photo_like(seed: int, size: int = 256) -> Image.Image:
    """Smooth colour fields with gradients, sensor noise and gamma.

    Uniform random noise is useless here: its histogram is already flat, so
    chi-square would flag it as stego before anything is embedded.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    channels = []
    for _ in range(3):
        field = gaussian_filter(rng.normal(size=(size, size)), 12)
        field = (field - field.min()) / (field.max() - field.min())
        base = 0.5 * field + 0.3 * xx + 0.2 * yy + rng.normal(scale=0.02, size=(size, size))
        channels.append(np.clip(base, 0, 1) ** (1 / 2.2) * 230 + 10)
    return Image.fromarray(np.round(np.stack(channels, -1)).astype(np.uint8))


def replace_lsbs(image: Image.Image, rate: float, sequential: bool, seed: int = 0) -> Image.Image:
    """Simulate LSB replacement of a random message on `rate` of all RGB values."""
    rng = np.random.default_rng(seed)
    values = np.array(image).reshape(-1)
    count = int(values.size * rate)
    idx = np.arange(count) if sequential else rng.choice(values.size, count, replace=False)
    values[idx] = (values[idx] & 0xFE) | rng.integers(0, 2, count, dtype=np.uint8)
    return Image.fromarray(values.reshape(np.array(image).shape))


PHOTOS = [photo_like(seed) for seed in range(3)]


# ---------------------------------------------------------------- bit plane


def test_bit_plane_extracts_requested_bit_and_channel():
    pixels = np.array([[[1, 0, 0], [2, 1, 0]], [[3, 0, 0], [4, 1, 0]]], dtype=np.uint8)
    image = Image.fromarray(pixels)
    assert np.array_equal(np.array(sa.bit_plane(image, "R", 0)), [[255, 0], [255, 0]])
    assert np.array_equal(np.array(sa.bit_plane(image, "R", 1)), [[0, 255], [255, 0]])
    assert np.array_equal(np.array(sa.bit_plane(image, "G", 0)), [[0, 255], [0, 255]])


@pytest.mark.parametrize("channel, bit", [("A", 0), ("R", 8), ("R", -1)])
def test_bit_plane_rejects_invalid_arguments(channel, bit):
    with pytest.raises(ValueError):
        sa.bit_plane(PHOTOS[0], channel, bit)


# ---------------------------------------------------------------- chi-square


@pytest.mark.parametrize("photo", PHOTOS)
def test_chi_square_does_not_flag_clean_photos(photo):
    assert sa.chi_square_probability(photo) < 0.1


@pytest.mark.parametrize("photo", PHOTOS)
def test_chi_square_flags_full_lsb_replacement(photo):
    assert sa.chi_square_probability(replace_lsbs(photo, 1.0, sequential=True)) > 0.99


@pytest.mark.parametrize("photo", PHOTOS)
def test_chi_square_curve_reveals_sequential_message_length(photo):
    curve = dict(sa.chi_square_curve(replace_lsbs(photo, 0.4, sequential=True), points=10))
    assert all(curve[f] > 0.99 for f in (0.1, 0.2, 0.3, 0.4))
    assert curve[1.0] < 0.95


# ---------------------------------------------------------------- RS analysis


@pytest.mark.parametrize("photo", PHOTOS)
def test_rs_reports_near_zero_for_clean_photos(photo):
    assert sa.rs_estimate(photo) < 0.06


@pytest.mark.parametrize("rate", [0.1, 0.3, 0.5, 0.7])
@pytest.mark.parametrize("photo", PHOTOS)
def test_rs_estimates_embedding_rate(photo, rate):
    estimate = sa.rs_estimate(replace_lsbs(photo, rate, sequential=False))
    assert estimate == pytest.approx(rate, abs=0.08)


@pytest.mark.parametrize("photo", PHOTOS)
def test_rs_does_not_collapse_at_full_embedding(photo):
    assert sa.rs_estimate(replace_lsbs(photo, 1.0, sequential=False)) > 0.95


# ---------------------------------------------------------------- our embedding


@pytest.mark.parametrize("photo", PHOTOS)
def test_lsb_matching_evades_both_attacks_at_full_capacity(photo):
    stego = lsb.embed(photo, "x" * lsb.capacity_bytes(photo), "pw")
    # Replacement at the same rate scores > 0.99 on both; matching stays far below.
    assert sa.chi_square_probability(stego) < 0.9
    assert sa.rs_estimate(stego) < 0.15
