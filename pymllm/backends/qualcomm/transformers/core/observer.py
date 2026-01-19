import torch
from torchao.quantization.pt2e import UniformQuantizationObserverBase


class ConcatObserver(UniformQuantizationObserverBase):
    """
    Fetch maximum data range of all tensors to be concatenated
    """

    def __init__(
        self,
        dtype=torch.uint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,  # noqa: B008
        is_dynamic=False,
        **kwargs,
    ) -> None:
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs,
        )

        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.register_buffer("min_val", torch.tensor(float("inf"), **factory_kwargs))
        self.register_buffer("max_val", torch.tensor(float("-inf"), **factory_kwargs))
        # get concat node and its inputs
        self.input_observers = []

    def add_observer(self, observer):
        self.input_observers.append(observer)

    def forward(self, x_orig):
        # calculate the min / max first
        self.min_val = min(self.min_val, x_orig.min())
        self.max_val = max(self.max_val, x_orig.max())

        # update min / max for all observers of input nodes
        for observers in self.input_observers:
            observers.min_val = self.min_val
            observers.max_val = self.max_val

        return x_orig

    def calculate_qparams(self):
        return self._calculate_qparams(self.min_val, self.max_val)
