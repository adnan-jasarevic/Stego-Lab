"""Password-based authenticated encryption for hidden payloads.

Blob layout: salt (16 B) | nonce (12 B) | AES-256-GCM ciphertext + tag (16 B).
"""

import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

SALT_SIZE = 16
NONCE_SIZE = 12
TAG_SIZE = 16
OVERHEAD = SALT_SIZE + NONCE_SIZE + TAG_SIZE

# Scrypt cost: ~50 ms per derivation, making password guessing expensive.
_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1


class DecryptionError(Exception):
    """Wrong password, or the hidden data was altered."""


def _derive_key(password: str, salt: bytes) -> bytes:
    kdf = Scrypt(salt=salt, length=32, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P)
    return kdf.derive(password.encode("utf-8"))


def encrypt(plaintext: bytes, password: str) -> bytes:
    salt = os.urandom(SALT_SIZE)
    nonce = os.urandom(NONCE_SIZE)
    ciphertext = AESGCM(_derive_key(password, salt)).encrypt(nonce, plaintext, None)
    return salt + nonce + ciphertext


def decrypt(blob: bytes, password: str) -> bytes:
    if len(blob) < OVERHEAD:
        raise DecryptionError("Hidden data is too short to be valid.")
    salt = blob[:SALT_SIZE]
    nonce = blob[SALT_SIZE : SALT_SIZE + NONCE_SIZE]
    ciphertext = blob[SALT_SIZE + NONCE_SIZE :]
    try:
        return AESGCM(_derive_key(password, salt)).decrypt(nonce, ciphertext, None)
    except InvalidTag:
        raise DecryptionError("Wrong password, or the image was modified.") from None
