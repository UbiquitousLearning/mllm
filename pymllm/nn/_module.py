# Copyright (c) MLLM Team.
# Licensed under the MIT License.

from .. import ffi
from ._layers import _Layer


class Module:
    def __init__(self):
        self.this_module_name: str = None
        self.absolute_name: str = None
        self.module_layer_list: list = []

    def load(self, pf: ffi.ParameterFile):
        for module_layer in self.module_layer_list:
            if isinstance(module_layer, Module, _Layer):
                module_layer.load(pf)
            else:
                raise TypeError(
                    "Module layer must be Module or _Layer, but got {}".format(
                        type(module_layer)
                    )
                )

    def trace(self):
        pass

    def forward(self, *args):
        # TODO send to engine's dispatcher
        pass

    def __call__(self, *args, **kwds):
        # __send_graph_begin()
        if kwds.get("__mllm_trace_mode_enabled", False):
            return self.trace(*args, **kwds)
        return self.forward(*args, **kwds)
        # __send_graph_end()
