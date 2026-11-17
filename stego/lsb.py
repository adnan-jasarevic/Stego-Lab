"""LSB matching steganography over password-scattered RGB values.

The R, G and B values of all pixels form one flat sequence. A password-derived
order picks which values carry the hidden bits, so the data is spread across
the whole image instead of starting at the top-left corner. Alpha is never
touched, so transparent areas stay identical.

When a value's lowest bit must change, it is randomly increased or decreased
by 1 (LSB matching) rather than having its bit overwritten (LSB replacement).
Replacement only ever turns even values up and odd values down, a pattern
chi-square and RS steganalysis detect; matching leaves no such asymmetry.

Layout of the embedded bit stream, in password order:

    MAGIC (4 B) | payload length (4 B, big-endian) | encrypted payload

The output must be saved losslessly (PNG); JPEG compression destroys the bits.
"""

import struct

import numpy as np
from PIL import Image

from . import crypto

MAGIC = b"STG2"
LEGACY_MAGIC = b"STG1"
HEADER = struct.Struct(">4sI")


class StegoError(Exception):
    pass


class MessageTooLarge(StegoError):
    pass


class NoMessageFound(StegoError):
    pass


class UnsupportedFormat(StegoError):
    pass


def _split_rgb(image: Image.Image) -> tuple[np.ndarray, Image.Image | None]:
    """Return the RGB pixels as uint8 array, plus the alpha band if present."""
    alpha = image.getchannel("A") if "A" in image.getbands() else None
    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    return rgb, alpha


def _merge_rgb(rgb: np.ndarray, alpha: Image.Image | None) -> Image.Image:
    out = Image.fromarray(rgb, "RGB")
    if alpha is not None:
        out.putalpha(alpha)
    return out


def _position_keys(password: str, count: int) -> np.ndarray:
    """One unique sort key per value index; the password order is ascending key.

    A 32-bit pseudo-random key goes in the high half and the index in the low
    half, so keys never tie and equal random parts fall back to index order.
    """
    random_part = np.frombuffer(crypto.position_keystream(password, count * 4), dtype=">u4")
    return (random_part.astype(np.uint64) << 32) | np.arange(count, dtype=np.uint64)


def _first_in_order(keys: np.ndarray, take: int) -> np.ndarray:
    """Indices of the `take` smallest keys, in key order.

    Selecting before sorting keeps this fast on large photos: sorting all
    36 M values of a 12 MP image takes seconds, selecting a few is linear.
    """
    if take >= keys.size:
        return np.argsort(keys)
    chosen = np.argpartition(keys, take)[:take]
    return chosen[np.argsort(keys[chosen])]


def positions(password: str, count: int, take: int | None = None) -> np.ndarray:
    """First `take` (default: all) of the `count` value indices in password order."""
    return _first_in_order(_position_keys(password, count), count if take is None else take)


def capacity_bytes(image: Image.Image) -> int:
    """Largest UTF-8 message size, in bytes, that fits in this image."""
    width, height = image.size
    return max(0, (width * height * 3) // 8 - HEADER.size - crypto.OVERHEAD)


def embed(image: Image.Image, message: str, password: str) -> Image.Image:
    plaintext = message.encode("utf-8")
    capacity = capacity_bytes(image)
    if len(plaintext) > capacity:
        raise MessageTooLarge(
            f"Message is {len(plaintext)} bytes; this image holds at most {capacity}."
        )

    payload = crypto.encrypt(plaintext, password)
    stream = HEADER.pack(MAGIC, len(payload)) + payload
    bits = np.unpackbits(np.frombuffer(stream, dtype=np.uint8))

    rgb, alpha = _split_rgb(image)
    flat = rgb.reshape(-1)
    targets = positions(password, flat.size, take=bits.size)

    values = flat[targets].astype(np.int16)
    mismatch = (values & 1) != bits
    step = np.random.default_rng().choice(np.array([-1, 1], dtype=np.int16), size=values.size)
    step[values == 0] = 1
    step[values == 255] = -1
    values[mismatch] += step[mismatch]
    flat[targets] = values.astype(np.uint8)
    return _merge_rgb(rgb, alpha)


def _bits_to_bytes(flat: np.ndarray, indices: np.ndarray) -> bytes:
    return np.packbits(flat[indices] & 1).tobytes()


def extract(image: Image.Image, password: str) -> str:
    rgb, _ = _split_rgb(image)
    flat = rgb.reshape(-1)
    available = flat.size // 8

    if available < HEADER.size:
        raise NoMessageFound("Image is too small to contain a message.")
    if _bits_to_bytes(flat, np.arange(32)) == LEGACY_MAGIC:
        raise UnsupportedFormat(
            "This image uses the old STG1 format, which is no longer supported."
        )

    keys = _position_keys(password, flat.size)
    header_bits = HEADER.size * 8
    magic, length = HEADER.unpack(_bits_to_bytes(flat, _first_in_order(keys, header_bits)))
    if magic != MAGIC or length > available - HEADER.size:
        raise NoMessageFound("No hidden message found, or the password is wrong.")

    order = _first_in_order(keys, header_bits + length * 8)
    payload = _bits_to_bytes(flat, order[header_bits:])
    return crypto.decrypt(payload, password).decode("utf-8")
