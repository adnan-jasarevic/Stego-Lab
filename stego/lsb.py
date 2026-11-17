"""LSB replacement steganography over the RGB channels of an image.

Bits are written sequentially into the least significant bit of each R, G, B
value (row-major order). Alpha is never touched, so transparent areas stay
identical. Layout of the embedded bit stream:

    MAGIC (4 B) | payload length (4 B, big-endian) | encrypted payload

The output must be saved losslessly (PNG); JPEG compression destroys the bits.
"""

import struct

import numpy as np
from PIL import Image

from . import crypto

MAGIC = b"STG1"
HEADER = struct.Struct(">4sI")


class StegoError(Exception):
    pass


class MessageTooLarge(StegoError):
    pass


class NoMessageFound(StegoError):
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
    flat[: bits.size] = (flat[: bits.size] & 0xFE) | bits
    return _merge_rgb(rgb, alpha)


def _read_bytes(flat: np.ndarray, start: int, count: int) -> bytes:
    bits = flat[start * 8 : (start + count) * 8] & 1
    return np.packbits(bits).tobytes()


def extract(image: Image.Image, password: str) -> str:
    rgb, _ = _split_rgb(image)
    flat = rgb.reshape(-1)
    available = flat.size // 8

    if available < HEADER.size:
        raise NoMessageFound("Image is too small to contain a message.")
    magic, length = HEADER.unpack(_read_bytes(flat, 0, HEADER.size))
    if magic != MAGIC or length > available - HEADER.size:
        raise NoMessageFound("No hidden message found in this image.")

    payload = _read_bytes(flat, HEADER.size, length)
    return crypto.decrypt(payload, password).decode("utf-8")
