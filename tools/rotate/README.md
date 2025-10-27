# RotLLM
This is an implementation of [SpinQuant](https://arxiv.org/abs/2405.16406) and [QuaRot](https://arxiv.org/abs/2404.00456) for different models like Qwen. We are not intented to do exactly the same things as SpinQuant and QuaRot, instead we provide a framework to customize rotation operations for any models you want to use.

![Example rotation for Qwen2](../../assets/rotation.png)

## Example
We provide a unified interface to rotate a model.
```python
import rotate
... # do whatever you want
rotate.rotate_model(model, ...) # parameters are customizable
```
You can find an example for `Qwen2ForCausalLM` and `Qwen2VLForConditionalGeneration` in [`qwen2.5-instruct.py`](./example/qwen2.5-instruct.py).

## WorkFlow of RotLLM
### Operations
The rotation operation on a model can be viewed as sequentially executing a series of predefined operations. Suppose you want to add a rotation operation for a model `abc`, first create `abc.py` in `rotate/model` and define operations as following
```python
from ..common import RotateOperationRegistry

# register the first step of operation to rotate model abc
@RotateOperationRegistry.register(abc)
def first_operation(model: abc, ...):
    ... # do whatever you want

@RotateOperationRegistry.register(abc)
def second_operation(model: abc, ...):
    ... # do whatever you want
```
After doing that, `rotate.rotate_model(model, ...)` will sequantially call `first_operation` and `second_operation` to handle model.

### Steps to rotate a model
#### Fuse layer norm
To ensure the invariance of a model, we should first fuse some operations of `norm` into the adjacent linear module.
Formally, 
```math
norm(x) = f(x) \circ w_n + b_n
```
in layer norm, we have
```math
f(x) = \frac{x-mean(x)}{\|x-mean(x)\|}
```
in RSM norm, we have
```math
f(x) = \frac{x}{\|x\|}
```
In LLMs, norm is usually followed by linear.
```math
\begin{aligned}
linear(norm(x)) &= norm(x)W_l + b_l \\
&=\left(f(x) \circ w_n + b_n \right)W_l + b_l \\
&=\left(f(x) diag(w_n) + b_n \right)W_l + b_l \\
&=f(x) \ diag(w_n)W_l + (b_nW_l + b_l)
\end{aligned}
```
This implies that $`norm(x)`$ is substitutable with $`f(x)`$. $`w_n`$ and $`b_n`$ can be fuse into linear layer
```math
\begin{aligned}
W_l &\rightarrow diag(w_n)W_l \\
b_n &\rightarrow b_nW_l + b_l
\end{aligned}
```

This is done by `fuse_layer_norms` in [rotatioin_utils.py](./rotate/rotation_utils.py).

The key problem is how `fuse_layer_norms` should identify the norm layers and their succeeding linear layers in diverse model architectures.

In our framework, to support a model like abc, you must implement a `NormLinearIterator` in abc.py, which iterates through the model and yields all `(father, norm_name, linears)` pairs. An example in [qwen.py](./rotate/model/qwen.py) is shown below
```python
from ..common import NormLinearIterator

@NormLinearIterator.register_iterator
class Qwen2NormLinearIterator(NormLinearIterator):
    def __init__(self, model: Qwen2ForCausalLM):
        super().__init__()
        self.model = model
        
    def __iter__(self):
        for layer in self.model.model.layers:
            yield layer, "input_layernorm", [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ]
            yield layer, "post_attention_layernorm", [
                layer.mlp.up_proj,
                layer.mlp.gate_proj,
            ]
        yield self.model.model, "norm", [self.model.lm_head]
        
    @classmethod
    def supports_model(cls, model: nn.Module) -> bool:
        return isinstance(model, Qwen2ForCausalLM) or isinstance(model, Qwen2VLForConditionalGeneration)
```

#### Rotate the model
The rotation operation on a model can be viewed as applying rotational transformations to either the inputs or outputs of certain layers while ensuring mathematical equivalence before and after rotation.

For different layer types (e.g., `embedding` and `linear`), the implementation of rotating their outputs varies. However, at an abstract level, both cases involve rotating outputs.

To streamline the code logic, our framework introduces the `AutoOperation` class, which encapsulates the same operation across different layers. This eliminates the need for conditional statements when applying the same operation to different layer types.

For details, you can refer to [common.py](./rotate/common.py) and [qwen.py](./rotate/model/qwen.py).

## Training rotation matrix
Currently, the rotation matrices we use are all random Hadamard matrices, which may not achieve optimal performance. According to SpinQuant, we can adopt a QAT (Quantization-Aware Training)-like approach to learn the rotation matrices for better results. This functionality has not yet been implemented and remains a TODO item.
