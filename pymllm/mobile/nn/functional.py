# Copyright (c) MLLM Team.
# Licensed under the MIT License.

import tvm_ffi
from .. import ffi

matmul_impl_default: int = ffi._ffi_api.matmul_impl_default()
matmul_impl_gguf: int = ffi._ffi_api.matmul_impl_gguf()
matmul_impl_blas: int = ffi._ffi_api.matmul_impl_blas()
matmul_impl_mllmblas: int = ffi._ffi_api.matmul_impl_mllmblas()


def matmul(
    lhs: ffi.Tensor,
    rhs: ffi.Tensor,
    transpose_lhs: bool = False,
    transpose_rhs: bool = False,
    impl: int = matmul_impl_default,
) -> ffi.Tensor:
    return tvm_ffi.get_global_func("mllm.nn.functional.matmul")(
        lhs, rhs, transpose_lhs, transpose_rhs, impl
    )
