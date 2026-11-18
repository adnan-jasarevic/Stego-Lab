# Stego Lab

Hide encrypted text in images, recover it with a password, and test images for hidden data.

- **Hide:** messages are encrypted with AES-256-GCM and written at password-chosen positions:
  - **PNG and other lossless images:** into pixel least significant bits, using LSB matching. The result is saved as PNG.
  - **JPEG:** into DCT coefficients, using F5 with matrix encoding. The result stays a JPEG with its original quality and metadata.
- **Measure:** MSE, PSNR and SSIM show how much the image changed.
- **Detect:** chi-square and RS steganalysis for pixels, JPEG calibration for F5, a row-by-row chi-square curve, and a bit-plane viewer.

## Quick start

Tested with Python 3.12.

```bash
git clone git@github.com:adnan-jasarevic/Stego-Lab.git
cd Stego-Lab
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

The app opens at <http://localhost:8501>.

| Tab | What it does |
|---|---|
| **Hide** | Upload a cover image, type a message and password, and download the result. A JPEG cover gives a JPEG, and anything else gives a PNG. A bar shows bytes used against capacity. After hiding, it shows MSE, PSNR and SSIM, plus a map of the pixels that changed. |
| **Reveal** | Upload a stego PNG or JPEG and enter the password to read the message. The format is detected from the file's contents. |
| **Analyse** | Upload any image to get a verdict from chi-square and RS analysis (plus JPEG calibration for JPEGs), the chi-square curve, and bit planes for each channel and bit. |

> **Never edit or re-save a stego image.** Converting a stego PNG to JPEG, editing or resizing it, or re-saving a stego JPEG destroys the hidden bits. So do apps that recompress images (most messengers and social networks). Send the file as a document or attachment instead.

## How hiding works

```
message ──UTF-8──▶ AES-256-GCM ──▶ "STG2" | length | salt | nonce | ciphertext | tag
                   (Scrypt key)                         │
                                                        ▼ one bit per value
password ──Scrypt──▶ AES-CTR keystream ──▶ order of all R, G, B values
                                                        │
                                                        ▼
                            LSB matching: value ±1 where its lowest bit differs
```

1. **Encryption:** the message is encrypted with AES-256-GCM, using a key derived from the password with Scrypt and a random salt and nonce. This adds 44 bytes. A wrong password or an altered image is detected rather than producing garbage.
2. **Positions:** a second, password-only keystream sorts all R, G and B values into a secret order. Bits go into the first values in that order, so they are spread over the whole image and the header has no fixed location. The order comes from AES-CTR, which is fixed by the AES standard, so images keep decoding after library upgrades.
3. **LSB matching:** when a value's lowest bit doesn't match the message bit, the value moves randomly by +1 or −1 (0 can only go up and 255 only down). Simple LSB *replacement* would only ever move even values up and odd values down, and that asymmetry is what classic detectors look for.
4. **Capacity:** width × height × 3 / 8 bytes, minus 52 bytes of header and encryption overhead. A 1920 × 1080 image holds about 777 KB. Alpha is never modified.

Without the password, the app can't tell an image with a message from one without, so a wrong password and "no message" produce the same response.

### JPEG mode (F5)

A JPEG stores each 8 × 8 block as quantised DCT coefficients. Decoding to pixels and re-encoding would re-quantise them and wipe out any hidden bits, so the app edits the coefficients directly with [`jpeglib`](https://pypi.org/project/jpeglib/).

1. **Same encryption and position order** as above. The header is `STJ1` | *k* | length, so a JPEG and a PNG message can never be confused.
2. **Which coefficients:** only non-zero AC coefficients carry bits. DC coefficients and zeros are skipped.
3. **Bit of a coefficient:** *c* mod 2 for positive *c*, and 1 − (|*c*| mod 2) for negative *c*. To flip a bit, |*c*| decreases by 1. If that turns it into 0 ("shrinkage"), the reader will skip it, so the same bits are embedded again further on.
4. **Matrix encoding:** *k* bits are carried by 2<sup>*k*</sup> − 1 coefficients with at most one change. *k* is chosen as large as the message allows, so short messages change very few coefficients.
5. **Capacity** depends on the image content and JPEG quality, so it is an estimate. It sits 4–7% below what actually fits, and in tests a message of exactly the reported capacity always fit. A 512 × 512 photo-like image at quality 85 holds about 2 KB. A detailed 12-megapixel photo at quality 90 holds about 530 KB.

On a 12 MP photo, a 100-byte message takes about 1 s and a 100 KB message about 7 s.

## Image quality

Each used value changes by at most 1. On a 256 × 256 image filled to capacity:

| MSE | PSNR | SSIM |
|---|---|---|
| 0.4994 | 51.15 dB | 0.99995 |

As a rough guide, PSNR above 40 dB is invisible to the eye. PSNR can't fall below 48.13 dB here, because every value moves by at most 1. High scores mean the image *looks* unchanged. They say nothing about whether a statistical test can detect the message.

In JPEG mode, one coefficient change alters a whole 8 × 8 block slightly. On 512 × 512 images at quality 85:

| Fill | PSNR | SSIM | File size |
|---|---|---|---|
| 10% of capacity | 53.2–53.8 dB | ≥ 0.997 | −2% |
| 100% of capacity | 44.1–44.3 dB | ≥ 0.974 | −20% |

## Reading the steganalysis results

| Result | Meaning |
|---|---|
| **Chi-square** | Probability that value pairs (2k, 2k+1) were evened out by LSB replacement. Clean photos score near 0%. |
| **RS estimate** | Estimated share of colour values carrying hidden bits. Within about ±8 points of the true rate from 10% to 70%. Above roughly 80% the standard formula breaks down, so a separate check reports full embedding instead of the false 0% the formula can produce. |
| **Chi-square curve** | Chi-square over the first *x*% of the image, read row by row. A message written from the start stays near 100% and then drops where the message ends. |
| **JPEG calibration** | JPEGs only. Fridrich, Goljan & Hogea (2002): crop 4 pixels, recompress with the same quantisation table, and compare low-frequency coefficient counts with the original. F5 leaves too many zeros and too few ±1s. The result is the estimated share of coefficients changed. |
| **Bit planes** | Natural images show shapes even in bit 0. Hidden data appears as static. |

The app flags an image when chi-square is above 90%, RS is above 10%, or JPEG calibration is above 10%. Calibration is only used at JPEG quality 70 and above, because heavy compression makes it unreliable: clean quality-50 test images reached 0.25.

**Measured on synthetic photo-like test images:**

| Image | Chi-square | RS |
|---|---|---|
| Clean | ≤ 0.03 | ≤ 0.04 |
| LSB replacement, full | > 0.99 | ≈ 1.00 |
| **This app's LSB matching, full capacity** | ≤ 0.45 | ≤ 0.06 |

Both detectors target LSB replacement, and the app's own PNG output gets past them. That doesn't make it undetectable. Modern machine-learning steganalysis (for example SRM features or CNN detectors) can detect LSB matching, especially at high capacity. Smaller messages are harder to find.

**JPEG calibration**, on JPEGs at quality 75–90:

| Image | Calibration estimate |
|---|---|
| Clean | ≤ 0.04 |
| This app's F5, 10% of capacity | 0.04–0.09 (true change rate 0.04–0.07), not reliably flagged |
| This app's F5, 40% of capacity | 0.17–0.28 (true 0.17–0.23), flagged |
| This app's F5, full capacity | 0.46–0.58 (true 0.46–0.49), flagged |

So the app's JPEG mode **is detectable** when a JPEG is filled heavily. Keep messages small relative to capacity.

## Limitations

- PNG output must stay lossless, and JPEG output must not be re-saved. Any recompression, resizing or editing destroys the message.
- JPEG mode loses capacity to shrinkage, especially in smooth images where most coefficients are ±1. The nsF5 variant avoids this with wet-paper codes but is not implemented.
- A stego JPEG is smaller than its cover (about 2% at light fill, 20% at full). Someone who has the original can notice the difference.
- The salt for the position order is fixed, because the receiver only has the password. Use a strong password.
- Detection results come from synthetic test images. Real photos vary, and heavily processed images can push chi-square up even when clean.

## Development

```bash
.venv/bin/python -m pytest -q
```

There are 104 tests covering PNG and JPEG embedding, encryption, metrics, steganalysis, and the app end to end (with simulated uploads). Expected values come from hand calculations or published constants, not from re-running the code under test. Each test module was also checked against deliberately broken versions of the code.

```
app.py                Streamlit UI
stego/crypto.py       Scrypt + AES-256-GCM, position keystream
stego/lsb.py          PNG mode: embed / extract / capacity
stego/jpeg.py         JPEG mode: F5 embed / extract / capacity, quality estimate
stego/metrics.py      MSE, PSNR, SSIM
stego/steganalysis.py bit planes, chi-square, RS analysis, JPEG calibration
tests/                pytest suite (images.py = synthetic test photos)
```

## License

[MIT](LICENSE)
