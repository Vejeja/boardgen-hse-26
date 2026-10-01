import hashlib
import os
from pathlib import Path


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def file_sha256(path: str | os.PathLike, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
