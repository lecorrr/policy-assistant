"""Password hashing and signed session tokens ("<login>.<expiry>.<signature>")."""

import base64
import hashlib
import hmac
import os
import re
import time

_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**14, 8, 1
_TOKEN = re.compile(r"^([a-z][a-z0-9._-]{1,31})\.(\d{1,12})\.([A-Za-z0-9_-]{43})$")


def hash_password(password: str) -> str:
    if len(password) < 8:
        raise ValueError("a password needs at least 8 characters")
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = _unb64(digest)
        actual = hashlib.scrypt(
            password.encode(), salt=_unb64(salt), n=int(n), r=int(r), p=int(p), dklen=len(expected)
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


class SessionSigner:
    def __init__(self, secret: str, max_age_s: float) -> None:
        if len(secret) < 32:
            raise ValueError("the session secret needs at least 32 characters")
        self._key = secret.encode()
        self.max_age_s = max_age_s

    def issue(self, login: str, now: float | None = None) -> str:
        expires = int((now if now is not None else time.time()) + self.max_age_s)
        payload = f"{login}.{expires}"
        return f"{payload}.{self._sign(payload)}"

    def login_of(self, token: str, now: float | None = None) -> str | None:
        """Return the login of a valid, unexpired token, else None."""
        match = _TOKEN.match(token or "")
        if match is None:
            return None
        login, expires, signature = match.groups()
        if not hmac.compare_digest(signature, self._sign(f"{login}.{expires}")):
            return None
        if int(expires) < (now if now is not None else time.time()):
            return None
        return login

    def _sign(self, payload: str) -> str:
        return _b64(hmac.new(self._key, payload.encode(), hashlib.sha256).digest())


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
