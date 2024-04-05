# Gemma LLM

## Arch

## Config

### 2B

```python
# The number of tokens in the vocabulary.
vocab_size: int = 256000
# The maximum sequence length that this model might ever be used with.
max_position_embeddings: int = 8192
# The number of blocks in the model.
num_hidden_layers: int = 18
# The number of attention heads used in the attention layers of the model.
num_attention_heads: int = 8
# The number of key-value heads for implementing attention.
num_key_value_heads: int = 1
# The hidden size of the model.
hidden_size: int = 2048
# The dimension of the MLP representations.
intermediate_size: int = 16384
# The number of head dimensions.
head_dim: int = 256
# The epsilon used by the rms normalization layers.
rms_norm_eps: float = 1e-6
```

### 7B

```python
# The number of tokens in the vocabulary.
vocab_size: int = 256000
# The maximum sequence length that this model might ever be used with.
max_position_embeddings: int = 8192
# The number of blocks in the model.
num_hidden_layers: int = 28
# The number of attention heads used in the attention layers of the model.
num_attention_heads: int = 16
# The number of key-value heads for implementing attention.
num_key_value_heads: int = 16
# The hidden size of the model.
hidden_size: int = 3072
# The dimension of the MLP representations.
intermediate_size: int = 24576
# The number of head dimensions.
head_dim: int = 256
# The epsilon used by the rms normalization layers.
rms_norm_eps: float = 1e-6
```

## Limits

## Licence