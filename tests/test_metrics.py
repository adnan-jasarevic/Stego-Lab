import math

import numpy as np
import pytest
from PIL import Image

from stego import lsb, metrics


def solid(value: int, size: int = 16) -> Image.Image:
    return Image.new("RGB", (size, size), (value, value, value))


@pytest.fixture
def cover() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8), "RGB")


def test_mse_of_single_changed_value():
    a = np.zeros((2, 2, 3), dtype=np.uint8)
    b = a.copy()
    b[0, 0, 1] = 4
    # One value differs by 4 among 2*2*3 = 12 values: 16 / 12.
    assert metrics.mse(Image.fromarray(a), Image.fromarray(b)) == pytest.approx(16 / 12)


def test_psnr_for_uniform_off_by_one():
    # MSE = 1 everywhere gives the textbook value 20*log10(255) = 48.1308 dB.
    assert metrics.psnr(solid(100), solid(101)) == pytest.approx(48.1308, abs=1e-4)


def test_identical_images_score_perfectly(cover):
    assert metrics.mse(cover, cover) == 0
    assert math.isinf(metrics.psnr(cover, cover))
    assert metrics.ssim(cover, cover) == pytest.approx(1.0)


def test_ssim_of_flat_images_matches_luminance_formula():
    # Zero variance leaves only the luminance term:
    # (2*100*110 + C1) / (100^2 + 110^2 + C1), with C1 = (0.01*255)^2 = 6.5025.
    expected = (22000 + 6.5025) / (22100 + 6.5025)
    assert metrics.ssim(solid(100), solid(110)) == pytest.approx(expected)


def test_ssim_scores_each_colour_channel_independently():
    # Only red differs (100 vs 110); green and blue are identical and score 1.
    a = Image.new("RGB", (16, 16), (100, 200, 50))
    b = Image.new("RGB", (16, 16), (110, 200, 50))
    red = (22000 + 6.5025) / (22100 + 6.5025)
    assert metrics.ssim(a, b) == pytest.approx((red + 1 + 1) / 3)


def test_ssim_penalises_structure_more_than_lsb_noise(cover):
    stego = lsb.embed(cover, "x" * (lsb.capacity_bytes(cover)), "pw")
    inverted = Image.fromarray(255 - np.asarray(cover))
    assert metrics.ssim(cover, stego) > 0.99
    assert metrics.ssim(cover, inverted) < 0


def test_full_capacity_lsb_stays_above_off_by_one_bound(cover):
    # Every value changes by at most 1, so MSE <= 1 and PSNR >= 48.13 dB.
    stego = lsb.embed(cover, "x" * lsb.capacity_bytes(cover), "pw")
    assert metrics.psnr(cover, stego) >= 48.1308


def test_alpha_differences_are_ignored():
    opaque = Image.new("RGBA", (8, 8), (50, 60, 70, 255))
    clear = Image.new("RGBA", (8, 8), (50, 60, 70, 0))
    assert metrics.mse(opaque, clear) == 0


def test_size_mismatch_is_rejected():
    with pytest.raises(ValueError):
        metrics.mse(solid(0, 8), solid(0, 9))
