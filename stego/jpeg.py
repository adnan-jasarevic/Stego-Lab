"""F5 steganography in the DCT coefficients of JPEG images.

JPEG stores each 8x8 block as quantised DCT coefficients. Hidden bits go into
the non-zero AC coefficients (DC and zeros are never used), visited in the
same password-derived order as the PNG mode. The image is never decoded to
pixels and re-encoded, so it keeps its original quality, EXIF and size.

Bit of a coefficient c:  c > 0 -> c mod 2,   c < 0 -> 1 - (|c| mod 2)
To flip a bit, |c| is decreased by 1 (towards zero). If that turns c into 0
("shrinkage"), the extractor will skip it, so the same bits are embedded again
in the following coefficients.

Matrix encoding (Westfeld, 2001): k message bits are carried by a group of
n = 2^k - 1 coefficients with at most one change. Larger k means fewer
changes per message bit, so k is chosen as large as the message size allows.

Layout, in password order:
    header with k = 1:  MAGIC (4 B) | k (1 B) | payload length (4 B)
    payload with k:     encrypted message (same format as the PNG mode)
"""

import os
import struct
import tempfile
from collections.abc import Iterator

import jpeglib
import numpy as np

from . import crypto
from .lsb import MessageTooLarge, NoMessageFound, _first_in_order, _position_keys

MAGIC = b"STJ1"
HEADER = struct.Struct(">4sBI")
MAX_K = 7
# Capacity is a statistical estimate (shrinkage depends on which bits must
# change); keep a margin so the reported capacity practically always fits.
_CAPACITY_MARGIN = 0.95


def is_jpeg(data: bytes) -> bool:
    return data[:3] == b"\xff\xd8\xff"


# ---------------------------------------------------------------- coefficients


def _components(jpeg: jpeglib.DCTJPEG) -> list[tuple[str, np.ndarray]]:
    comps = [(name, getattr(jpeg, name)) for name in ("Y", "Cb", "Cr")]
    return [(name, arr) for name, arr in comps if arr is not None]


def _ac_coefficients(jpeg: jpeglib.DCTJPEG) -> np.ndarray:
    """All AC coefficients of all components as one flat int64 array."""
    return np.concatenate(
        [arr.reshape(-1, 64)[:, 1:].reshape(-1) for _, arr in _components(jpeg)]
    ).astype(np.int64)


def _store_ac(jpeg: jpeglib.DCTJPEG, ac: np.ndarray) -> None:
    offset = 0
    for name, arr in _components(jpeg):
        blocks = arr.reshape(-1, 64).copy()
        count = blocks.shape[0] * 63
        blocks[:, 1:] = ac[offset : offset + count].reshape(-1, 63)
        offset += count
        setattr(jpeg, name, blocks.reshape(arr.shape).astype(arr.dtype))


def _read(data: bytes, workdir: str) -> jpeglib.DCTJPEG:
    path = os.path.join(workdir, "in.jpg")
    with open(path, "wb") as f:
        f.write(data)
    jpeg = jpeglib.read_dct(path)
    _components(jpeg)  # jpeglib loads lazily; load while the file exists
    return jpeg


class _Walk:
    """Coefficients in password order, skipping zeros, extended on demand.

    Sorting every coefficient of a large photo is slow, so the order is
    computed in growing prefixes (same result as one full sort).
    """

    def __init__(self, ac: np.ndarray, password: str):
        self.ac = ac
        self._keys = _position_keys(password, ac.size)
        self._ordered = 0
        self.indices: list[int] = []
        self.values: list[int] = []
        self.pos = 0

    def _extend(self) -> bool:
        if self._ordered >= self.ac.size:
            return False
        take = min(self.ac.size, max(4096, 2 * self._ordered))
        new = _first_in_order(self._keys, take)[self._ordered :]
        new = new[self.ac[new] != 0]
        self.indices.extend(new.tolist())
        self.values.extend(self.ac[new].tolist())
        self._ordered = take
        return True

    def group(self, size: int) -> list[int] | None:
        """Positions (into self.values) of the next `size` non-zero coefficients."""
        found, p = [], self.pos
        while len(found) < size:
            if p >= len(self.values) and not self._extend():
                return None
            if self.values[p] != 0:
                found.append(p)
            p += 1
        return found

    def commit(self) -> None:
        self.ac[np.array(self.indices, dtype=np.int64)] = self.values


def _bit(value: int) -> int:
    return value & 1 if value > 0 else 1 - (value & 1)


def _syndrome(values: list[int], group: list[int]) -> int:
    h = 0
    for i, p in enumerate(group, start=1):
        if _bit(values[p]):
            h ^= i
    return h


# ---------------------------------------------------------------- F5


class _OutOfSpace(Exception):
    pass


def _embed_chunks(walk: _Walk, chunks: Iterator[int], k: int) -> None:
    n = (1 << k) - 1
    values = walk.values
    for m in chunks:
        while True:
            group = walk.group(n)
            if group is None:
                raise _OutOfSpace
            s = _syndrome(values, group) ^ m
            if s:
                p = group[s - 1]
                values[p] += -1 if values[p] > 0 else 1
                if values[p] == 0:
                    continue  # shrinkage: retry with the next coefficients
            walk.pos = group[-1] + 1
            break


def _read_chunks(walk: _Walk, count: int, k: int) -> list[int] | None:
    n = (1 << k) - 1
    chunks = []
    for _ in range(count):
        group = walk.group(n)
        if group is None:
            return None
        chunks.append(_syndrome(walk.values, group))
        walk.pos = group[-1] + 1
    return chunks


def _to_chunks(data: bytes, k: int) -> list[int]:
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    bits = np.concatenate([bits, np.zeros(-bits.size % k, dtype=np.uint8)])
    weights = 1 << np.arange(k - 1, -1, -1)
    return (bits.reshape(-1, k) * weights).sum(axis=1).tolist()


def _from_chunks(chunks: list[int], k: int, length: int) -> bytes:
    bits = ((np.array(chunks)[:, None] >> np.arange(k - 1, -1, -1)) & 1).reshape(-1)
    return np.packbits(bits[: length * 8].astype(np.uint8)).tobytes()


# ---------------------------------------------------------------- capacity


def _cost_per_chunk(k: int, ones_share: float) -> float:
    """Expected coefficients consumed per k-bit chunk, including shrinkage."""
    n = (1 << k) - 1
    change = n / (n + 1)
    return n + change * ones_share / max(1e-9, 1 - change * ones_share)


def _usable(ac: np.ndarray) -> tuple[float, float]:
    """Non-zero AC coefficients left after the header, and the share of +/-1."""
    nonzero = int(np.count_nonzero(ac))
    ones_share = float(np.count_nonzero(np.abs(ac) == 1)) / max(1, nonzero)
    header = HEADER.size * 8 * _cost_per_chunk(1, ones_share)
    return (nonzero - header) * _CAPACITY_MARGIN, ones_share


def _payload_capacity(ac: np.ndarray) -> int:
    usable, ones_share = _usable(ac)
    bits = usable / _cost_per_chunk(1, ones_share)
    return max(0, int(bits) // 8 - crypto.OVERHEAD)


def capacity_bytes(data: bytes) -> int:
    """Approximate largest UTF-8 message size, in bytes, for this JPEG."""
    with tempfile.TemporaryDirectory() as tmp:
        return _payload_capacity(_ac_coefficients(_read(data, tmp)))


def _k_candidates(ac: np.ndarray, payload_bits: int) -> list[int]:
    """Largest k whose estimated cost fits first, then smaller ones as fallbacks."""
    usable, ones_share = _usable(ac)
    best = 1
    for k in range(1, MAX_K + 1):
        if -(-payload_bits // k) * _cost_per_chunk(k, ones_share) <= usable:
            best = k
    return list(range(best, 0, -1))


# ---------------------------------------------------------------- public API


def embed(data: bytes, message: str, password: str) -> bytes:
    if not is_jpeg(data):
        raise ValueError("Not a JPEG file.")
    plaintext = message.encode("utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        jpeg = _read(data, tmp)
        ac = _ac_coefficients(jpeg)
        capacity = _payload_capacity(ac)
        if len(plaintext) > capacity:
            raise MessageTooLarge(
                f"Message is {len(plaintext)} bytes; this JPEG holds about {capacity}."
            )

        payload = crypto.encrypt(plaintext, password)
        for k in _k_candidates(ac, len(payload) * 8):
            walk = _Walk(ac.copy(), password)
            try:
                _embed_chunks(walk, iter(_to_chunks(HEADER.pack(MAGIC, k, len(payload)), 1)), 1)
                _embed_chunks(walk, iter(_to_chunks(payload, k)), k)
            except _OutOfSpace:
                continue
            walk.commit()
            _store_ac(jpeg, walk.ac)
            out = os.path.join(tmp, "out.jpg")
            jpeg.write_dct(out)
            with open(out, "rb") as f:
                return f.read()
    raise MessageTooLarge("Message does not fit in this JPEG; shorten it or use a larger image.")


def extract(data: bytes, password: str) -> str:
    if not is_jpeg(data):
        raise ValueError("Not a JPEG file.")
    with tempfile.TemporaryDirectory() as tmp:
        ac = _ac_coefficients(_read(data, tmp))

    walk = _Walk(ac, password)
    header_chunks = _read_chunks(walk, HEADER.size * 8, 1)
    if header_chunks is None:
        raise NoMessageFound("Image is too small to contain a message.")
    magic, k, length = HEADER.unpack(_from_chunks(header_chunks, 1, HEADER.size))
    if magic != MAGIC or not 1 <= k <= MAX_K or length > np.count_nonzero(ac) // 8:
        raise NoMessageFound("No hidden message found, or the password is wrong.")

    chunks = _read_chunks(walk, -(-length * 8 // k), k)
    if chunks is None:
        raise NoMessageFound("No hidden message found, or the password is wrong.")
    return crypto.decrypt(_from_chunks(chunks, k, length), password).decode("utf-8")


# ---------------------------------------------------------------- quality

# Base luminance quantisation table, JPEG standard (ITU T.81) Annex K.
_BASE_LUMINANCE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
])


def _ijg_table(quality: int) -> np.ndarray:
    """Luminance table libjpeg (and Pillow) use for a quality setting of 1-100."""
    scale = 5000 // quality if quality < 50 else 200 - 2 * quality
    return np.clip((_BASE_LUMINANCE * scale + 50) // 100, 1, 255)


def estimate_quality(data: bytes) -> int:
    """Closest standard quality setting (1-100) for this JPEG's luminance table.

    Cameras and phones often use their own tables; the result is then the
    standard setting with the most similar overall compression.
    """
    with tempfile.TemporaryDirectory() as tmp:
        jpeg = _read(data, tmp)
        luminance = jpeg.qt[jpeg.quant_tbl_no[0]].astype(np.int64)
    errors = [np.abs(_ijg_table(q) - luminance).sum() for q in range(1, 101)]
    return int(np.argmin(errors)) + 1
