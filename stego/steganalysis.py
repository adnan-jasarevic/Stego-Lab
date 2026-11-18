"""Detectors that look for hidden LSB data in an image.

- bit_plane: visualise one bit of one channel; hidden data looks like noise.
- chi_square_*: Westfeld & Pfitzmann (1999). LSB replacement evens out the
  counts of each value pair (2k, 2k+1); the test gives the probability that
  this has happened.
- rs_estimate: Fridrich, Goljan & Du (2001). Estimates the fraction of values
  that carry hidden bits from how flipping LSBs changes local smoothness.
- jpeg_calibration_estimate: Fridrich, Goljan & Hogea (2002). Estimates the
  share of DCT coefficients changed by F5-style JPEG embedding.

Both statistical attacks target LSB replacement. LSB matching (used by
stego.lsb) does not create the pair symmetry they rely on.
"""

import numpy as np
from PIL import Image
from scipy.stats import chi2

CHANNELS = ("R", "G", "B")


def _rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.int64)


def bit_plane(image: Image.Image, channel: str = "R", bit: int = 0) -> Image.Image:
    """Black-and-white image of one bit of one colour channel."""
    if channel not in CHANNELS:
        raise ValueError(f"Channel must be one of {CHANNELS}.")
    if not 0 <= bit <= 7:
        raise ValueError("Bit must be between 0 and 7.")
    values = _rgb(image)[..., CHANNELS.index(channel)]
    return Image.fromarray((((values >> bit) & 1) * 255).astype(np.uint8), "L")


# ---------------------------------------------------------------- chi-square

_MIN_EXPECTED = 5  # chi-square is unreliable for pairs with fewer samples


def _chi_square_p(values: np.ndarray) -> float:
    hist = np.bincount(values.reshape(-1), minlength=256).astype(np.float64)
    even, odd = hist[0::2], hist[1::2]
    expected = (even + odd) / 2
    used = expected >= _MIN_EXPECTED
    if used.sum() < 2:
        return 0.0
    statistic = np.sum((even[used] - expected[used]) ** 2 / expected[used])
    return float(chi2.sf(statistic, df=used.sum() - 1))


def chi_square_probability(image: Image.Image) -> float:
    """Probability (0-1) that the whole image's LSBs were replaced with data."""
    return _chi_square_p(_rgb(image))


def chi_square_curve(image: Image.Image, points: int = 20) -> list[tuple[float, float]]:
    """Probability over growing prefixes of the image, read row by row.

    Returns (fraction of image scanned, probability) pairs. With sequential
    embedding the probability stays near 1 until the end of the message and
    then drops, revealing its length.
    """
    flat = _rgb(image).reshape(-1)
    curve = []
    for i in range(1, points + 1):
        fraction = i / points
        curve.append((fraction, _chi_square_p(flat[: max(1, int(flat.size * fraction))])))
    return curve


# ---------------------------------------------------------------- RS analysis

_RS_MASK = np.array([0, 1, 1, 0])
_RS_SATURATION = 0.1


def _flip_positive(x: np.ndarray) -> np.ndarray:
    """F1: 0<->1, 2<->3, ... (LSB flip)."""
    return x ^ 1


def _flip_negative(x: np.ndarray) -> np.ndarray:
    """F-1: -1<->0, 1<->2, 3<->4, ..."""
    return ((x + 1) ^ 1) - 1


def _smoothness(groups: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(groups, axis=-1)).sum(axis=-1)


def _regular_singular(groups: np.ndarray, flip) -> tuple[float, float]:
    """Fractions of groups that get rougher (regular) or smoother (singular)."""
    flipped = groups.copy()
    masked = _RS_MASK.astype(bool)
    flipped[:, masked] = flip(groups[:, masked])
    before, after = _smoothness(groups), _smoothness(flipped)
    return float(np.mean(after > before)), float(np.mean(after < before))


def _rs_channel(values: np.ndarray) -> float:
    width = values.shape[1] - values.shape[1] % _RS_MASK.size
    groups = values[:, :width].reshape(-1, _RS_MASK.size)
    inverted = _flip_positive(groups)

    r_m, s_m = _regular_singular(groups, _flip_positive)
    r_nm, s_nm = _regular_singular(groups, _flip_negative)
    r_m1, s_m1 = _regular_singular(inverted, _flip_positive)
    r_nm1, s_nm1 = _regular_singular(inverted, _flip_negative)

    d0, d1 = r_m - s_m, r_m1 - s_m1
    dn0, dn1 = r_nm - s_nm, r_nm1 - s_nm1

    # Embedding drives d0 towards 0 while dn0 grows (in clean images d0 ~ dn0).
    # Near full embedding the quadratic below becomes unstable and can even
    # return 0, so treat a near-zero d0 as "LSBs fully randomised".
    if dn0 > 0 and d0 < _RS_SATURATION * dn0:
        return 1.0

    a = 2 * (d1 + d0)
    b = dn0 - dn1 - d1 - 3 * d0
    c = d0 - dn0

    if abs(a) < 1e-12:
        x = -c / b if abs(b) > 1e-12 else 0.0
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            return 0.0
        roots = ((-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a))
        x = min(roots, key=abs)
    if abs(x - 0.5) < 1e-12:
        return 1.0
    return float(np.clip(x / (x - 0.5), 0.0, 1.0))


def rs_estimate(image: Image.Image) -> float:
    """Estimated fraction (0-1) of RGB values carrying hidden bits, averaged over channels."""
    rgb = _rgb(image)
    return float(np.mean([_rs_channel(rgb[..., c]) for c in range(3)]))


# ---------------------------------------------------------------- JPEG calibration

# Low-frequency DCT modes used by the attack: (row, column) inside each 8x8 block.
_CALIBRATION_MODES = ((0, 1), (1, 0), (1, 1))


def _abs_histogram(blocks: np.ndarray, mode: tuple[int, int]) -> np.ndarray:
    """Share of blocks whose coefficient at `mode` has |value| = 0, 1, 2."""
    values = np.abs(blocks[..., mode[0], mode[1]].reshape(-1))
    return np.array([np.count_nonzero(values == d) for d in (0, 1, 2)]) / values.size


def _calibrated_luminance(path: str, workdir: str) -> tuple[np.ndarray, np.ndarray]:
    """Luminance DCT blocks of the image and of its cropped, recompressed version."""
    import jpeglib

    original = jpeglib.read_dct(path)
    luminance_qt = original.qt[original.quant_tbl_no[0]]
    gray = jpeglib.read_spatial(path, out_color_space=jpeglib.JCS_GRAYSCALE).spatial[..., 0]

    cropped = np.ascontiguousarray(gray[4:, 4:, None])

    recompressed = f"{workdir}/calibrated.jpg"
    jpeglib.from_spatial(
        cropped, in_color_space=jpeglib.JCS_GRAYSCALE
    ).write_spatial(recompressed, qt=luminance_qt[None])
    return original.Y, jpeglib.read_dct(recompressed).Y


def jpeg_calibration_estimate(data: bytes) -> float:
    """Estimated share (0-1) of non-zero AC coefficients changed by F5-style embedding.

    Fridrich, Goljan & Hogea (2002): cropping 4 pixels and recompressing with
    the same quantisation table yields an image whose coefficient histogram
    approximates the cover's. F5 moves coefficients towards zero, so the stego
    image has too many 0s and too few 1s compared with that estimate.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as workdir:
        path = f"{workdir}/input.jpg"
        with open(path, "wb") as f:
            f.write(data)
        stego, calibrated = _calibrated_luminance(path, workdir)

    estimates = []
    for mode in _CALIBRATION_MODES:
        h = _abs_histogram(stego, mode)
        c = _abs_histogram(calibrated, mode)
        denominator = c[1] ** 2 + (c[2] - c[1]) ** 2
        if denominator > 0:
            estimates.append((c[1] * (h[0] - c[0]) + (h[1] - c[1]) * (c[2] - c[1])) / denominator)
    return float(np.clip(np.mean(estimates), 0.0, 1.0)) if estimates else 0.0
