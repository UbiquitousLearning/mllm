from .base import _LIB
from . import _ffi_api


def echo(rec: str) -> None:
    return _ffi_api.echo(rec)
