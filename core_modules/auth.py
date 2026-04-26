import base64
import datetime
import hashlib
import hmac
import logging
import os
from typing import Dict, Optional

from jose import ExpiredSignatureError, JWTError, jwt

from core_modules.config import Config

logger = logging.getLogger(__name__)


def hash_password(password: str, iterations: Optional[int] = None) -> str:
    config = Config()
    rounds = iterations or config.PASSWORD_HASH_ITERATIONS
    salt = base64.b64encode(os.urandom(16)).decode("utf-8")
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        rounds,
    )
    encoded = base64.b64encode(digest).decode("utf-8")
    return f"pbkdf2_sha256${rounds}${salt}${encoded}"


def _verify_password(stored_password: str, provided_password: str) -> bool:
    if not stored_password:
        return False

    if stored_password.startswith("pbkdf2_sha256$"):
        try:
            _, rounds, salt, encoded_hash = stored_password.split("$", 3)
            derived = hashlib.pbkdf2_hmac(
                "sha256",
                provided_password.encode("utf-8"),
                salt.encode("utf-8"),
                int(rounds),
            )
            expected = base64.b64decode(encoded_hash.encode("utf-8"))
            return hmac.compare_digest(derived, expected)
        except Exception as exc:
            logger.warning("PBKDF2 password verification failed: %s", exc)
            return False

    legacy_hash = hashlib.sha256(provided_password.encode("utf-8")).hexdigest()
    return hmac.compare_digest(stored_password, legacy_hash)


class Authentication:
    def __init__(self, db=None, config: Optional[Config] = None):
        self.db = db
        self.config = config or Config()
        self.secret_key = self.config.SECRET_KEY

    def _hash_password(self, password: str) -> str:
        return hash_password(password, self.config.PASSWORD_HASH_ITERATIONS)

    def authenticate(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user against DB; fallback to configured demo user."""
        if self.db:
            try:
                user = self.db.get_user(username)
                if user and _verify_password(user["password"], password):
                    return {k: user[k] for k in ("id", "username", "name", "email", "role")}
            except Exception as exc:
                logger.warning("DB auth error: %s", exc)

        if (
            username == self.config.DEMO_ADMIN_USERNAME
            and password == self.config.DEMO_ADMIN_PASSWORD
        ):
            return {
                "id": 1,
                "username": self.config.DEMO_ADMIN_USERNAME,
                "name": self.config.DEMO_ADMIN_NAME,
                "email": self.config.DEMO_ADMIN_EMAIL,
                "role": "admin",
            }
        return None

    def generate_token(self, user_id: int) -> str:
        payload = {
            "user_id": user_id,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=self.config.JWT_EXPIRY_HOURS),
            "iat": datetime.datetime.utcnow(),
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    def verify_token(self, token: str) -> Optional[Dict]:
        try:
            return jwt.decode(token, self.secret_key, algorithms=["HS256"])
        except ExpiredSignatureError:
            logger.warning("JWT token expired")
        except JWTError as exc:
            logger.warning("Invalid JWT: %s", exc)
        return None
