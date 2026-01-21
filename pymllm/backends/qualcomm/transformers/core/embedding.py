import torch
import torch.nn as nn
from torch.ao.quantization import FakeQuantize, MinMaxObserver


class QEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        quant_bits=16,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.quant_bits = quant_bits

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight)

        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].fill_(0)

        # Quantization configuration for Weight
        self.weight_fake_quant = FakeQuantize(
            observer=MinMaxObserver.with_args(
                qscheme=torch.per_tensor_affine,
                dtype=torch.qint32,
                eps=0.0001 / 65535,
            ),
            quant_min=0,
            quant_max=2 ** (quant_bits) - 1,
            dtype=torch.qint32,
            qscheme=torch.per_tensor_affine,
        )

    def forward(self, x):
        # 1. Weight fake quantization
        # If observer is not closed, this step will continuously update scale/zp
        # If freeze_weight() is called, this will just use fixed scale/zp for quantization
        w_q = self.weight_fake_quant(self.weight)

        # 2. Embedding lookup (Gather operation)
        return nn.functional.embedding(
            x,
            w_q,
            padding_idx=self.padding_idx,
        )

    @torch.no_grad()
    def convert_to_deploy(self):
        """
        In-place replacement of self.weight:
        Float Parameter -> Int Buffer
        """
        # 1. Ensure quantization parameters are ready
        if self.weight_fake_quant.scale is None:
            self.freeze_weight()

        scale = self.weight_fake_quant.scale
        zero_point = self.weight_fake_quant.zero_point
        quant_min = self.weight_fake_quant.quant_min
        quant_max = self.weight_fake_quant.quant_max

        # 2. Calculate integer values
        # w_int = round(w / s + zp)
        w_int = torch.round(self.weight / scale + zero_point).clamp(
            quant_min, quant_max
        )

        # 3. Set target integer type
        if self.quant_bits <= 8:
            target_dtype = torch.uint8
        elif self.quant_bits <= 16:
            target_dtype = torch.uint16
        else:
            target_dtype = torch.uint32

        w_int = w_int.to(target_dtype)

        # === Key steps: Replacement operations ===

        # A. Delete original Parameter 'weight'
        # Must delete first, otherwise cannot register buffer with same name
        del self.weight

        # B. Register Buffer with same name 'weight'
        # This makes state_dict['weight'] become Int Tensor
        self.register_buffer("weight", w_int)

        # C. Register Scale (usually needed by engine)
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)

        # D. Clean up unnecessary modules
        if hasattr(self, "weight_fake_quant"):
            del self.weight_fake_quant

        class_name = self.__class__.__name__
        instance_class_name = type(self).__name__
        print(
            f"Class: {class_name}, Instance: {instance_class_name}, Deploy Mode Activated. 'weight' is now {self.weight.dtype} buffer. zp is {zero_point}"
        )

    @torch.no_grad()
    def freeze_weight(self):
        """
        Manually trigger Observer to observe and calculate scale, then lock it.
        Solve the problem of output being 0 on first run.
        """
        self.weight_fake_quant.activation_post_process(self.weight)
        s, zp = self.weight_fake_quant.activation_post_process.calculate_qparams()
        self.weight_fake_quant.scale.copy_(s)
        self.weight_fake_quant.zero_point.copy_(zp)
        self.weight_fake_quant.disable_observer()
        class_name = self.__class__.__name__
        instance_class_name = type(self).__name__
        print(
            f"Class: {class_name}, Instance: {instance_class_name}, Weight Quantized: scale={self.weight_fake_quant.scale}, zp={self.weight_fake_quant.zero_point}"
        )

    def disable_quant(self):
        """Completely turn off quantization noise and return to floating point mode"""
        self.weight_fake_quant.disable_fakequant()

    def extra_repr(self):
        s = f"{self.num_embeddings}, {self.embedding_dim}"
        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"
        return s
