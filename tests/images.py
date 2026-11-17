"""Synthetic test images shared by the test modules."""

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


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
