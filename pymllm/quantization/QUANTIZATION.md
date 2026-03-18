# pymllm Quantization Guide

## Architecture

pymllm uses a **plugin-based** quantization system. Each quantization
algorithm (AWQ, GPTQ, FP8, W8A8, ...) is a self-contained plugin that
implements three methods: **create weights**, **apply** (forward), and
**process weights after loading**.

```
                          QuantizationConfig
                          (parses checkpoint)
                                 │
                                 │  get_quant_method(layer, prefix)
                                 ▼
┌─────────────────────────────────────────────────────┐
│               LinearMethodBase                      │
│                                                     │
│  create_weights()   ← called during layer __init__  │
│  apply()            ← called during layer forward   │
│  process_weights_after_loading()  ← called once     │
│                       after checkpoint is loaded     │
└─────────────────────────────────────────────────────┘
                                 │
                                 │  registered on layer as
                                 │  layer.quant_method
                                 ▼
                    Linear / ColumnParallelLinear / ...
```

### Key modules

| Module | Purpose |
|--------|---------|
| `pymllm.layers.quantize_base` | `QuantizeMethodBase`, `LinearMethodBase`, `UnquantizedLinearMethod` |
| `pymllm.quantization.quant_config` | `QuantizationConfig` base class, registry, factory |
| `pymllm.quantization.methods/` | Concrete implementations (AWQ, GPTQ, FP8, ...) |

## Lifecycle

### 1. Model construction

Each linear layer accepts an optional `quant_method` argument. If `None`,
`UnquantizedLinearMethod` is used (standard FP weight + `F.linear`).

```python
from pymllm.layers.linear import ColumnParallelLinear

# No quantization (default)
layer = ColumnParallelLinear(4096, 4096)

# With quantization
layer = ColumnParallelLinear(4096, 4096, quant_method=my_quant_method)
```

During `__init__`, the layer calls:

```python
self.quant_method.create_weights(
    layer=self,
    input_size_per_partition=in_features,
    output_partition_sizes=[out_features_per_partition],
    input_size=in_features,
    output_size=out_features,
    params_dtype=torch.get_default_dtype(),
    weight_loader=self.weight_loader,
)
```

This registers the appropriate parameters on the layer. For unquantized
layers, this is a single `weight` parameter. For AWQ, it might be
`qweight` (packed int32), `scales` (fp16), and `qzeros` (packed int32).

### 2. Weight loading

The standard `model.load_weights(iter)` loop loads checkpoint tensors into
the parameters created above, using each parameter's `weight_loader`
attribute for tensor-parallel sharding.

### 3. Post-load processing

After all weights are loaded, `ModelRunner` calls:

```python
for name, module in model.named_modules():
    quant_method = getattr(module, "quant_method", None)
    if quant_method is not None:
        quant_method.process_weights_after_loading(module)
```

This is where format conversions happen:
- **AWQ**: repack AutoAWQ int4 layout → Marlin kernel layout
- **GPTQ**: shuffle weights according to `g_idx` for exllama kernels
- **FP8**: quantize FP16 weights to FP8 and compute per-tensor scales

### 4. Inference

Every forward call goes through `quant_method.apply()`:

```python
# Inside ColumnParallelLinear.forward():
output = self.quant_method.apply(self, x, self.bias)
```

For unquantized layers this is just `F.linear`. For quantized layers it
invokes a fused dequant+matmul kernel.

## How to add a new quantization method

### Step 1: Implement `LinearMethodBase`

Create a file in `pymllm/quantization/methods/`, e.g. `awq.py`:

```python
from pymllm.layers.quantize_base import LinearMethodBase
from pymllm.layers.utils import set_weight_attrs

class AWQLinearMethod(LinearMethodBase):
    \"\"\"AWQ W4A16 quantized linear method.\"\"\"

    def __init__(self, weight_bits: int, group_size: int, zero_point: bool):
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.pack_factor = 32 // weight_bits  # e.g. 8 for 4-bit

    def create_weights(
        self, layer, input_size_per_partition, output_partition_sizes,
        input_size, output_size, params_dtype, **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)

        # Packed 4-bit weights: each int32 holds 8 x 4-bit values
        qweight = Parameter(
            torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(qweight, {"input_dim": 0, "output_dim": 1})
        layer.register_parameter("qweight", qweight)
        set_weight_attrs(qweight, extra_weight_attrs)

        # Per-group scales
        scales = Parameter(
            torch.empty(
                input_size_per_partition // self.group_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("scales", scales)
        set_weight_attrs(scales, extra_weight_attrs)

        # Per-group zero-points (packed)
        qzeros = Parameter(
            torch.empty(
                input_size_per_partition // self.group_size,
                output_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("qzeros", qzeros)
        set_weight_attrs(qzeros, extra_weight_attrs)

    def apply(self, layer, x, bias=None):
        # Dequantize and compute matmul
        # In practice, call a fused CUDA kernel here
        out = awq_dequantize_and_gemm(x, layer.qweight, layer.scales, layer.qzeros)
        if bias is not None:
            out = out + bias
        return out

    def process_weights_after_loading(self, layer):
        # Optional: repack weights for a faster kernel layout
        # e.g. convert AutoAWQ format → Marlin format
        layer.qweight = Parameter(layer.qweight.data, requires_grad=False)
        layer.scales = Parameter(layer.scales.data, requires_grad=False)
        layer.qzeros = Parameter(layer.qzeros.data, requires_grad=False)
```

### Step 2: Implement `QuantizationConfig`

```python
from pymllm.quantization.quant_config import QuantizationConfig, register_quantization

@register_quantization("awq")
class AWQConfig(QuantizationConfig):
    def __init__(self, weight_bits, group_size, zero_point):
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point

    def get_name(self) -> str:
        return "awq"

    @classmethod
    def from_config(cls, config: dict) -> "AWQConfig":
        return cls(
            weight_bits=config["bits"],
            group_size=config["group_size"],
            zero_point=config["zero_point"],
        )

    def get_quant_method(self, layer, prefix=""):
        # Skip quantization for certain layers if needed
        # if "lm_head" in prefix:
        #     return None
        return AWQLinearMethod(self.weight_bits, self.group_size, self.zero_point)
```

### Step 3: Use it

```python
from pymllm.quantization import get_quantization_config

# Parse from checkpoint config
ConfigClass = get_quantization_config("awq")
config = ConfigClass.from_config({"bits": 4, "group_size": 128, "zero_point": True})

# Create layer with quantization
quant_method = config.get_quant_method(layer=None, prefix="model.layers.0.q_proj")
layer = ColumnParallelLinear(4096, 4096, quant_method=quant_method)
```

## API Reference

### `QuantizeMethodBase`

| Method | When called | Purpose |
|--------|-------------|---------|
| `create_weights(layer, ...)` | `layer.__init__` | Register parameters (weight, scales, etc.) on the layer |
| `apply(layer, x, bias)` | `layer.forward` | Quantized matmul computation |
| `process_weights_after_loading(layer)` | After `load_weights` | Repack / transform loaded checkpoint tensors |

### `QuantizationConfig`

| Method | Purpose |
|--------|---------|
| `get_name()` | Return method name (e.g. `"awq"`) |
| `from_config(config_dict)` | Class method: parse checkpoint JSON into config instance |
| `get_quant_method(layer, prefix)` | Return `LinearMethodBase` for a specific layer |
| `get_supported_act_dtypes()` | Activation dtypes this method supports |
| `get_min_capability()` | Minimum CUDA compute capability |
| `get_config_filenames()` | Checkpoint files to probe (default: `["quantize_config.json"]`) |

### Registry functions

| Function | Purpose |
|----------|---------|
| `@register_quantization("name")` | Decorator to register a config class |
| `get_quantization_config("name")` | Look up registered config class by name |
| `list_quantization_methods()` | List all registered method names |
