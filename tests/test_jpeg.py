import io
import tempfile

import numpy as np
import pytest
from PIL import Image

from stego import jpeg
from stego.crypto import DecryptionError
from tests.images import photo_like


def to_jpeg(image: Image.Image, **options) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", **options)
    return buf.getvalue()


def blocks(data: bytes) -> dict[str, np.ndarray]:
    """All DCT blocks per component, as (n_blocks, 64) int arrays."""
    with tempfile.TemporaryDirectory() as tmp:
        parsed = jpeg._read(data, tmp)
        return {name: arr.reshape(-1, 64).astype(int) for name, arr in jpeg._components(parsed)}


def header_k(data: bytes, password: str) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        walk = jpeg._Walk(jpeg._ac_coefficients(jpeg._read(data, tmp)), password)
    chunks = jpeg._read_chunks(walk, jpeg.HEADER.size * 8, 1)
    return jpeg.HEADER.unpack(jpeg._from_chunks(chunks, 1, jpeg.HEADER.size))[1]


@pytest.fixture(scope="module")
def cover() -> bytes:
    return to_jpeg(photo_like(0, size=384), quality=85)


VARIANTS = {
    "4:2:0 q75": dict(quality=75),
    "4:4:4 q95": dict(quality=95, subsampling=0),
    "progressive q85": dict(quality=85, progressive=True),
}


@pytest.mark.parametrize("options", VARIANTS.values(), ids=VARIANTS.keys())
def test_short_and_full_capacity_messages_survive(options):
    data = to_jpeg(photo_like(1, size=384), **options)
    short = "Meet at dawn — čćšđž 🌅"
    full = "x" * jpeg.capacity_bytes(data)
    assert jpeg.extract(jpeg.embed(data, short, "pw"), "pw") == short
    assert jpeg.extract(jpeg.embed(data, full, "pw"), "pw") == full


def test_grayscale_jpeg_works():
    data = to_jpeg(photo_like(2, size=384).convert("L"), quality=85)
    assert jpeg.extract(jpeg.embed(data, "grey", "pw"), "pw") == "grey"


@pytest.mark.parametrize("seed", range(4))
def test_reported_capacity_always_fits(seed):
    data = to_jpeg(photo_like(seed + 10, size=256), quality=80)
    message = "y" * jpeg.capacity_bytes(data)
    for password in ("a", "b", "c"):
        assert jpeg.extract(jpeg.embed(data, message, password), password) == message


def test_message_far_above_capacity_is_refused(cover):
    with pytest.raises(jpeg.MessageTooLarge):
        jpeg.embed(cover, "x" * (2 * jpeg.capacity_bytes(cover)), "pw")


def test_wrong_password_finds_nothing(cover):
    stego = jpeg.embed(cover, "secret", "right")
    with pytest.raises(jpeg.NoMessageFound):
        jpeg.extract(stego, "wrong")


def test_clean_jpeg_reports_no_message(cover):
    with pytest.raises(jpeg.NoMessageFound):
        jpeg.extract(cover, "pw")


def test_altered_payload_coefficient_is_detected(cover):
    stego = jpeg.embed(cover, "secret", "pw")
    k = header_k(stego, "pw")
    with tempfile.TemporaryDirectory() as tmp:
        parsed = jpeg._read(stego, tmp)
        ac = jpeg._ac_coefficients(parsed)
        walk = jpeg._Walk(ac, "pw")
        jpeg._read_chunks(walk, jpeg.HEADER.size * 8, 1)
        target = walk.indices[walk.group((1 << k) - 1)[0]]
        ac[target] += 1 if ac[target] > 0 else -1  # flips its bit, stays non-zero
        jpeg._store_ac(parsed, ac)
        parsed.write_dct(f"{tmp}/tampered.jpg")
        tampered = open(f"{tmp}/tampered.jpg", "rb").read()
    with pytest.raises(DecryptionError):
        jpeg.extract(tampered, "pw")


def test_only_ac_magnitudes_shrink_by_one(cover):
    before = blocks(cover)
    after = blocks(jpeg.embed(cover, "x" * 500, "pw"))
    for name in before:
        b, a = before[name], after[name]
        assert np.array_equal(a[:, 0], b[:, 0]), f"{name}: DC changed"
        changed = a != b
        assert changed.any()
        assert np.all(np.abs(a[changed]) == np.abs(b[changed]) - 1)
        assert np.all(a[changed] * b[changed] >= 0), f"{name}: sign flipped"


def test_short_messages_use_matrix_encoding(cover):
    capacity = jpeg.capacity_bytes(cover)
    assert header_k(jpeg.embed(cover, "x" * 20, "pw"), "pw") >= 4
    assert header_k(jpeg.embed(cover, "x" * capacity, "pw"), "pw") == 1


def test_stego_keeps_quality_size_and_exif():
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90° clockwise when displayed
    data = to_jpeg(photo_like(3, size=384), quality=90, exif=exif)
    stego = jpeg.embed(data, "hello", "pw")
    image = Image.open(io.BytesIO(stego))

    assert image.format == "JPEG"
    assert image.size == (384, 384)
    assert image.getexif().get(0x0112) == 6
    assert jpeg.estimate_quality(stego) == 90


@pytest.mark.parametrize("quality", [10, 50, 69, 70, 85, 100])
def test_quality_estimate_matches_save_setting(quality):
    assert jpeg.estimate_quality(to_jpeg(photo_like(0, size=64), quality=quality)) == quality


def test_non_jpeg_input_is_rejected():
    png = io.BytesIO()
    photo_like(0, size=64).save(png, format="PNG")
    with pytest.raises(ValueError):
        jpeg.embed(png.getvalue(), "hi", "pw")
