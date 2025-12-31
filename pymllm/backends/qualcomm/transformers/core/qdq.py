import torch
import torch.nn as nn
from torch.ao.quantization import FakeQuantize, MinMaxObserver


class ActivationQDQInt16PerTensorSym(nn.Module):
    def __init__(self):
        super().__init__()
        self.fake_quant = FakeQuantize(
            observer=MinMaxObserver,
            quant_min=-32768,
            quant_max=32767,
            dtype=torch.qint32,
            qscheme=torch.per_tensor_symmetric,
        )
        self.enable_observer()

    def forward(self, x):
        return self.fake_quant(x)

    def enable_observer(self):
        self.fake_quant.enable_observer()

    def disable_observer(self):
        self.fake_quant.disable_observer()


class ActivationQDQInt8PerTensorSym(nn.Module):
    def __init__(self):
        super().__init__()
        self.fake_quant = FakeQuantize(
            observer=MinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint32,
            qscheme=torch.per_tensor_symmetric,
        )
        self.enable_observer()

    def forward(self, x):
        return self.fake_quant(x)

    def enable_observer(self):
        self.fake_quant.enable_observer()

    def disable_observer(self):
        self.fake_quant.disable_observer()


QDQ_OP = {
    "A8-PerTensor": ActivationQDQInt8PerTensorSym,
    "A16-PerTensor": ActivationQDQInt16PerTensorSym,
}
