"""FP32 cuBLAS SGEMM on Numba-owned device memory and the SAME CUDA stream.

The installed NVIDIA CUDA toolkit supplies the DLL. No tensor cores, no TF32,
no NumPy/CPU multiplication. Geometry inside cuBLAS belongs to NVIDIA's backend.
"""
import ctypes as ct
import os
from pathlib import Path


class Blas:
    def __init__(self, stream):
        root = Path(os.environ["CUDA_HOME"])
        self.path = root / "bin/x64/cublas64_13.dll"
        self.dll = ct.WinDLL(str(self.path))
        self.handle = ct.c_void_p()
        self._signature("cublasCreate_v2", [ct.POINTER(ct.c_void_p)])
        self._signature("cublasSetStream_v2", [ct.c_void_p, ct.c_void_p])
        self._signature("cublasSetMathMode", [ct.c_void_p, ct.c_int])
        self._signature("cublasSetAtomicsMode", [ct.c_void_p, ct.c_int])
        self._signature("cublasGetVersion_v2", [ct.c_void_p, ct.POINTER(ct.c_int)])
        self._signature("cublasDestroy_v2", [ct.c_void_p])
        self._signature("cublasSgemm_v2", [ct.c_void_p, ct.c_int, ct.c_int,
            ct.c_int, ct.c_int, ct.c_int, ct.POINTER(ct.c_float), ct.c_void_p, ct.c_int,
            ct.c_void_p, ct.c_int, ct.POINTER(ct.c_float), ct.c_void_p, ct.c_int])
        self._call("cublasCreate_v2", ct.byref(self.handle))
        raw = stream.handle
        self._call("cublasSetStream_v2", self.handle, ct.c_void_p(raw.value if hasattr(raw, "value") else int(raw)))
        self._call("cublasSetMathMode", self.handle, 2)  # CUBLAS_PEDANTIC_MATH
        self._call("cublasSetAtomicsMode", self.handle, 0)  # CUBLAS_ATOMICS_NOT_ALLOWED
        version = ct.c_int()
        self._call("cublasGetVersion_v2", self.handle, ct.byref(version))
        self.version = version.value

    def _signature(self, name, args):
        fn = getattr(self.dll, name)
        fn.argtypes, fn.restype = args, ct.c_int

    def _call(self, name, *args):
        status = getattr(self.dll, name)(*args)
        if status:
            raise RuntimeError(f"{name}: cuBLAS status {status}")

    @staticmethod
    def ptr(array):
        return ct.c_void_p(array.device_ctypes_pointer.value)

    def gemm(self, a, b, out, M, N, K, transpose_a=False, transpose_b=False):
        """C_row=op(A)_row @ op(B)_row, implemented as C_col=B_col @ A_col."""
        alpha, beta = ct.c_float(1.), ct.c_float(0.)
        self._call("cublasSgemm_v2", self.handle, int(transpose_b), int(transpose_a),
                   N, M, K, ct.byref(alpha), self.ptr(b), K if transpose_b else N,
                   self.ptr(a), M if transpose_a else K, ct.byref(beta), self.ptr(out), N)

    def close(self):
        if self.handle:
            self._call("cublasDestroy_v2", self.handle)
            self.handle = ct.c_void_p()
