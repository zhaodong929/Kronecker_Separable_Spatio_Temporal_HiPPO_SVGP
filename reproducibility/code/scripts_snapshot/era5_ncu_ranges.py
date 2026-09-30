"""Small CUDA NVTX helper used by the cross-framework FLOP profiler."""

from __future__ import annotations

import ctypes
import os
from typing import Any, Optional


_NVTX: Optional[Any] = None
_PROFILE_CLAIMED = False


def _library() -> Optional[Any]:
    global _NVTX
    if _NVTX is not None:
        return _NVTX
    if not os.environ.get("ERA5_NCU_PROFILE_RANGE"):
        return None
    for name in ("libnvToolsExt.so.1", "libnvToolsExt.so"):
        try:
            library = ctypes.CDLL(name)
        except OSError:
            continue
        library.nvtxRangePushA.argtypes = [ctypes.c_char_p]
        library.nvtxRangePushA.restype = ctypes.c_int
        library.nvtxRangePop.argtypes = []
        library.nvtxRangePop.restype = ctypes.c_int
        _NVTX = library
        return library
    _NVTX = False
    return None


def profile_this_index(index: int, total: int) -> bool:
    """Return whether this is the single range to be profiled in this process."""

    global _PROFILE_CLAIMED
    if _PROFILE_CLAIMED or _library() is None:
        return False
    target = os.environ.get("ERA5_NCU_PROFILE_TARGET", "last").strip().lower()
    selected = target == "all" or (target == "last" and index == total - 1)
    if target.isdigit():
        selected = index == int(target)
    if selected:
        _PROFILE_CLAIMED = True
    return selected


def push_range(name: str, active: bool) -> bool:
    library = _library() if active else None
    if library is None:
        return False
    library.nvtxRangePushA(name.encode("utf-8"))
    return True


def pop_range(active: bool) -> None:
    library = _library() if active else None
    if library is not None:
        library.nvtxRangePop()
