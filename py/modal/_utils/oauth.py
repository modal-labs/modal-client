# Copyright Modal Labs 2026
import base64
import hashlib
import json
import time
import uuid
from collections.abc import Mapping
from typing import Any

from modal.exception import InvalidError

_OAUTH_CLIENT_ASSERTION_AUDIENCE = "modal-server"
_OAUTH_CLIENT_ASSERTION_LIFETIME_SECONDS = 5 * 60
_MIN_RSA_KEY_SIZE = 2048


class OAuthAssertionProvider:
    """Mint short-lived private_key_jwt assertions for OAuth client authentication."""

    def __init__(self, client_id: str, private_key_pem: str):
        try:
            import jwt
            from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
            from jwt.algorithms import RSAAlgorithm
            from jwt.exceptions import InvalidKeyError
        except ImportError:
            raise ImportError(
                "Private-key OAuth authentication requires the 'oauth' extra. "
                "Install it with `pip install 'modal[oauth]'`."
            ) from None

        private_key_pem = private_key_pem.replace("\\n", "\n")
        try:
            private_key = RSAAlgorithm(RSAAlgorithm.SHA256).prepare_key(private_key_pem)
        except (TypeError, InvalidKeyError):
            raise InvalidError("oauth_jwt_key must be an unencrypted RSA private key encoded as PEM.") from None
        if not isinstance(private_key, RSAPrivateKey):
            raise InvalidError("oauth_jwt_key must be an RSA private key.")
        if private_key.key_size < _MIN_RSA_KEY_SIZE:
            raise InvalidError("oauth_jwt_key must be at least 2048 bits.")

        public_jwk: dict[str, Any] = RSAAlgorithm.to_jwk(private_key.public_key(), as_dict=True)
        thumbprint_jwk = {field: public_jwk[field] for field in ("e", "kty", "n")}
        canonical_jwk = json.dumps(thumbprint_jwk, sort_keys=True, separators=(",", ":")).encode()

        self._client_id = client_id
        self._private_key = private_key
        self._kid = base64.urlsafe_b64encode(hashlib.sha256(canonical_jwk).digest()).rstrip(b"=").decode()
        self._jwt = jwt

    def __call__(self) -> Mapping[str, str]:
        issued_at = int(time.time())
        assertion = self._jwt.encode(
            {
                "iss": self._client_id,
                "sub": self._client_id,
                "aud": _OAUTH_CLIENT_ASSERTION_AUDIENCE,
                "iat": issued_at,
                "exp": issued_at + _OAUTH_CLIENT_ASSERTION_LIFETIME_SECONDS,
                "jti": str(uuid.uuid4()),
            },
            self._private_key,
            algorithm="RS256",
            headers={"kid": self._kid},
        )
        return {"x-modal-oauth-client-assertion": assertion}
