# Copyright (c) MLLM Team.
# Licensed under the MIT License.

import tvm_ffi
from .. import ffi


class _Layer:
    def __init__(self):
        self.device: ffi.Device = ffi.cpu_()
        self.this_layer_name: str = None
        self.absolute_name: str = None
        self._mllm_c_op_ptr: ffi.BaseOp = None
        self._params_file_ptr: ffi.ParameterFile = None

    def load(self, pf: ffi.ParameterFile):
        self._mllm_c_op_ptr.load(pf)
        self._params_file_ptr = pf

    def trace(self):
        pass

    def forward(self, *args):
        inputs = []
        for arg in args:
            if isinstance(arg, (ffi.Tensor)):
                inputs.append(arg)
            else:
                print(
                    f"The layer's forward function received a none Tensor type of {type(arg)}. Which is not supported."
                )
        ret = tvm_ffi.get_global_func("mllm.engine.dispatch")(
            self.device, self._mllm_c_op_ptr, inputs
        )
        if len(ret) == 1:
            return ret[0]
        return ret

    def __call__(self, *args, **kwds):
        return self.forward(*args)

    def __repr__(self):
        return "_Layer"


class Linear(_Layer):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "nn.Linear"


class Softmax(_Layer):
    def __init__(
        self,
        dim=-1,
    ):
        super().__init__()
        self.dim = dim
        self._mllm_c_op_ptr = ffi.SoftmaxOp.create(
            self.device, ffi.SoftmaxOpOptions(dim)
        )

    def __repr__(self):
        return f"mllm.aops.Softmax(dim={self.dim})"


class RoPE(_Layer):
    def __init__(self):
        super().__init__()
