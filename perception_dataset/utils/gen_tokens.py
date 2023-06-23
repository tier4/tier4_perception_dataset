from secrets import token_bytes, token_hex, token_urlsafe
from typing import Any


def generate_token(nbytes: int = 16, mode: str = "hex") -> Any:
    if nbytes < 16:
        raise ValueError(f"nbytes {nbytes} is too short. Give >= 16.")
    if mode == "bytes":
        return token_bytes(nbytes=nbytes)
    elif mode == "hex":
        return token_hex(nbytes=nbytes)
    elif mode == "urlsafe":
        return token_urlsafe(nbytes=nbytes)
    else:
        raise ValueError(f"Invalid argument 'mode'='{mode}'")
