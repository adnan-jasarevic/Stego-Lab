import numpy as np
import pytest
from PIL import Image

from stego import lsb
from stego import steganalysis as sa
from tests.images import photo_like, replace_lsbs

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


# ---------------------------------------------------------------- JPEG calibration


def _jpeg(seed: int, quality: int) -> bytes:
    import io

    buf = io.BytesIO()
    photo_like(seed, size=512).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _low_frequency_coefficients(data: bytes) -> np.ndarray:
    import tempfile

    from stego import jpeg

    with tempfile.TemporaryDirectory() as tmp:
        y = jpeg._read(data, tmp).Y
    return np.stack([y[..., r, c].reshape(-1) for r, c in sa._CALIBRATION_MODES])


@pytest.mark.parametrize("quality", [75, 90])
@pytest.mark.parametrize("seed", range(3))
def test_calibration_stays_low_on_clean_jpegs(seed, quality):
    assert sa.jpeg_calibration_estimate(_jpeg(seed, quality)) < 0.06


@pytest.mark.parametrize("fill", [0.4, 1.0])
@pytest.mark.parametrize("seed", range(3))
def test_calibration_tracks_true_f5_change_rate(seed, fill):
    from stego import jpeg

    cover = _jpeg(seed, 80)
    stego = jpeg.embed(cover, "x" * int(jpeg.capacity_bytes(cover) * fill), "pw")
    before, after = _low_frequency_coefficients(cover), _low_frequency_coefficients(stego)
    true_rate = np.count_nonzero(after != before) / np.count_nonzero(before)

    estimate = sa.jpeg_calibration_estimate(stego)
    assert estimate == pytest.approx(true_rate, abs=0.1)
    assert estimate > 0.1
