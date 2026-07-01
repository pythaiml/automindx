# services/secrets.py
# Secrets management (codephreak audit #10). Resolve secrets from, in order:
#   1. the OS keyring (via the optional `keyring` lib / HashiCorp Vault backend)
#   2. an environment variable
# Never hard-coded, never logged in full. If keyring isn't installed we degrade
# to env vars — automindX still runs, just without the keyring hardening.
from __future__ import annotations

import os
from typing import Optional

SERVICE = os.getenv("AUTOMINDX_KEYRING_SERVICE", "automindx")


def get_secret(name: str, service: str = SERVICE) -> Optional[str]:
    """Return a secret by name (keyring first, then env). None if unset."""
    if not name:
        return None
    try:
        import keyring  # optional dependency
        val = keyring.get_password(service, name)
        if val:
            return val
    except Exception:
        pass  # keyring unavailable → fall through to env
    return os.getenv(name) or os.getenv(name.upper()) or None


def set_secret(name: str, value: str, service: str = SERVICE) -> bool:
    """Store a secret in the OS keyring (never on disk in plaintext)."""
    try:
        import keyring
        keyring.set_password(service, name, value)
        return True
    except Exception:
        return False


def redact(value: Optional[str]) -> str:
    """Safe-to-log representation of a secret (never the full value)."""
    if not value:
        return "<unset>"
    if len(value) <= 6:
        return "***"
    return f"{value[:3]}…{value[-2:]} (len {len(value)})"
