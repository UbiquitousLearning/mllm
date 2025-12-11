# Copyright (c) MLLM Team.
# Licensed under the MIT License.

from .. import ffi


class _Layer:
    def __init__(self):
        self.this_layer_name: str = None
        self.absolute_name: str = None
        self._mllm_c_op_ptr: ffi.BaseOp = None
        self._params_file_ptr: ffi.ParameterFile = None

    def load(self, pf: ffi.ParameterFile):
        self._mllm_c_op_ptr.load(pf)
        self._params_file_ptr = pf

    def trace(self):
        pass

    def forward(self):
        # TODO dispatch op
        pass

    def __call__(self, *args, **kwds):
        pass


class Linear(_Layer):
    def __init__(self):
        super().__init__()


class Softmax(_Layer):
    def __init__(self):
        super().__init__()


class RoPE(_Layer):
    def __init__(self):
        super().__init__()
