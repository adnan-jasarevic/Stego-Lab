"""Image-quality metrics comparing a cover image with its stego version.

All metrics use the RGB channels only (embedding never changes alpha) and
treat pixel values on the 0-255 scale.

Rough guide for 8-bit images:
    PSNR > 40 dB  -> difference invisible to the eye; LSB typically scores ~51 dB
    SSIM > 0.99   -> structurally indistinguishable
High scores say nothing about statistical detectability; see steganalysis.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

MAX_VALUE = 255.0

# Constants from Wang et al., "Image Quality Assessment: From Error
# Visibility to Structural Similarity" (2004).
_SSIM_SIGMA = 1.5
_SSIM_C1 = (0.01 * MAX_VALUE) ** 2
_SSIM_C2 = (0.03 * MAX_VALUE) ** 2


def _rgb_pair(cover: Image.Image, stego: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    if cover.size != stego.size:
        raise ValueError(f"Image sizes differ: {cover.size} vs {stego.size}.")
    a = np.asarray(cover.convert("RGB"), dtype=np.float64)
    b = np.asarray(stego.convert("RGB"), dtype=np.float64)
    return a, b


def mse(cover: Image.Image, stego: Image.Image) -> float:
    a, b = _rgb_pair(cover, stego)
    return float(np.mean((a - b) ** 2))


def psnr(cover: Image.Image, stego: Image.Image) -> float:
    """Peak signal-to-noise ratio in dB; infinite for identical images."""
    error = mse(cover, stego)
    if error == 0:
        return float("inf")
    return float(10 * np.log10(MAX_VALUE**2 / error))


def ssim(cover: Image.Image, stego: Image.Image) -> float:
    """Mean structural similarity over all pixels and RGB channels (1.0 = identical).

    Uses an 11x11 Gaussian window (sigma 1.5) per channel, as in Wang et al.
    """
    a, b = _rgb_pair(cover, stego)

    def blur(x: np.ndarray) -> np.ndarray:
        # Smooth spatially only; never mix colour channels.
        return gaussian_filter(x, sigma=(_SSIM_SIGMA, _SSIM_SIGMA, 0), truncate=3.5)

    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a**2
    var_b = blur(b * b) - mu_b**2
    cov = blur(a * b) - mu_a * mu_b

    numerator = (2 * mu_a * mu_b + _SSIM_C1) * (2 * cov + _SSIM_C2)
    denominator = (mu_a**2 + mu_b**2 + _SSIM_C1) * (var_a + var_b + _SSIM_C2)
    return float(np.mean(numerator / denominator))
