from _typeshed import Incomplete
from pathlib import Path

_DEFAULT_KEY: Incomplete
MAGIC: bytes
VERSION: int

def _get_key(key: bytes = None) -> bytes: ...
def encrypt_file(src_path: str | Path, dst_path: str | Path = None, key: bytes = None) -> str: ...
def decrypt_file(path: str | Path, key: bytes = None) -> bytes: ...
