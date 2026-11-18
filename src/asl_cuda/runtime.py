"""Local toolkit discovery. Never installs packages or silently falls back to CPU."""
import os
from pathlib import Path

_DLL_HANDLES = []


def configure():
    if os.environ.get("NUMBA_ENABLE_CUDASIM", "0") != "0":
        raise RuntimeError("CUDA simulator is not a GPU benchmark.")
    if os.name == "nt":
        root = Path(os.environ.get("CUDA_HOME", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1"))
        os.environ.setdefault("CUDA_HOME", str(root))
        for folder in (root / "bin/x64", root / "bin", root / "nvvm/bin/x64", root / "nvvm/bin"):
            if folder.is_dir():
                _DLL_HANDLES.append(os.add_dll_directory(str(folder)))


configure()
# numba-cuda 0.30.4 still registers NumPy's retired row_stack alias at import.
# Preserve the installed NumPy 2.5.5: the alias is exactly vstack, not new math.
# Process-local compatibility only; no site-packages files are changed.
import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
from numba import cuda  # noqa: E402


def device_info():
    if not cuda.is_available():
        raise RuntimeError("A real CUDA device is required.")
    d = cuda.get_current_device()
    free, total = cuda.current_context().get_memory_info()
    return dict(name=d.name.decode() if isinstance(d.name, bytes) else d.name,
                compute_capability=list(d.compute_capability), free_bytes=int(free), total_bytes=int(total),
                max_threads_per_block=int(d.MAX_THREADS_PER_BLOCK),
                max_shared_memory_per_block=int(d.MAX_SHARED_MEMORY_PER_BLOCK),
                simulator=False)
