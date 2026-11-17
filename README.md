# Stego Lab

Hide encrypted text in images, recover it with a password, and test images for hidden data.

- **Hide:** messages are encrypted with AES-256-GCM and written into the least significant bits of password-chosen pixel values, using LSB matching.
- **Measure:** MSE, PSNR and SSIM show how much the image changed.
- **Detect:** chi-square and RS steganalysis, a row-by-row chi-square curve, and a bit-plane viewer.

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
| **Hide** | Upload a cover image, type a message and password, and download the result as PNG. A bar shows bytes used against capacity. After hiding, it shows MSE, PSNR and SSIM, plus a map of the pixels that changed. |
| **Reveal** | Upload a stego PNG and enter the password to read the message. |
| **Analyse** | Upload any image to get a verdict from chi-square and RS analysis, the chi-square curve, and bit planes for each channel and bit. |

> **Always keep stego images as PNG.** JPEG compression, and apps that recompress images (most messengers and social networks), destroy the hidden bits. JPEG support is tracked in [#6](https://github.com/adnan-jasarevic/Stego-Lab/issues/6).

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

## Image quality

Each used value changes by at most 1. On a 256 × 256 image filled to capacity:

| MSE | PSNR | SSIM |
|---|---|---|
| 0.4994 | 51.15 dB | 0.99995 |

As a rough guide, PSNR above 40 dB is invisible to the eye. PSNR can't fall below 48.13 dB here, because every value moves by at most 1. High scores mean the image *looks* unchanged. They say nothing about whether a statistical test can detect the message.

## Reading the steganalysis results

| Result | Meaning |
|---|---|
| **Chi-square** | Probability that value pairs (2k, 2k+1) were evened out by LSB replacement. Clean photos score near 0%. |
| **RS estimate** | Estimated share of colour values carrying hidden bits. Within about ±8 points of the true rate from 10% to 70%. Above roughly 80% the standard formula breaks down, so a separate check reports full embedding instead of the false 0% the formula can produce. |
| **Chi-square curve** | Chi-square over the first *x*% of the image, read row by row. A message written from the start stays near 100% and then drops where the message ends. |
| **Bit planes** | Natural images show shapes even in bit 0. Hidden data appears as static. |

The app flags an image when chi-square is above 90% or RS is above 10%.

**Measured on synthetic photo-like test images:**

| Image | Chi-square | RS |
|---|---|---|
| Clean | ≤ 0.03 | ≤ 0.04 |
| LSB replacement, full | > 0.99 | ≈ 1.00 |
| **This app's LSB matching, full capacity** | ≤ 0.45 | ≤ 0.06 |

Both detectors target LSB replacement, and the app's own output gets past them. That doesn't make it undetectable. Modern machine-learning steganalysis (for example SRM features or CNN detectors) can detect LSB matching, especially at high capacity. Smaller messages are harder to find.

## Limitations

- Output must stay lossless (PNG). Any recompression, resizing or editing destroys the message.
- The salt for the position order is fixed, because the receiver only has the password. Use a strong password.
- Detection results come from synthetic test images. Real photos vary, and heavily processed images can push chi-square up even when clean.

## Development

```bash
.venv/bin/python -m pytest -q
```

There are 67 tests covering embedding, encryption, metrics, steganalysis, and the app end to end (with simulated uploads). Expected values come from hand calculations or published constants, not from re-running the code under test. Each test module was also checked against deliberately broken versions of the code.

```
app.py                Streamlit UI
stego/crypto.py       Scrypt + AES-256-GCM, position keystream
stego/lsb.py          embed / extract / capacity
stego/metrics.py      MSE, PSNR, SSIM
stego/steganalysis.py bit planes, chi-square, RS analysis
tests/                pytest suite (images.py = synthetic test photos)
```
