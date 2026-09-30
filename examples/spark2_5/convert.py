# Copyright (c) MLLM Team.
# Licensed under the MIT License.
"""Validate the official Spark 1.7B inventory and emit a float32 ModelFileV2.

The tied output projection is materialized from model.embedding.weight. This
portable conversion needs torch and safetensors, not a compiled mllm binding.
"""
import argparse
import json
import os
from pathlib import Path
import struct

import torch
from safetensors import safe_open


def expected_shapes(config):
    c = config
    if (c['model_type'], c['hidden_size'], c['num_hidden_layers'], c['intermediate_size'],
        c['num_attention_heads'], c['num_key_value_heads'], c['head_dim'], c['vocab_size']) != (
        'spark2_5', 2048, 28, 6656, 8, 2, 256, 131072):
        raise ValueError('Only the official Spark-X2.5-1.7B geometry is supported')
    required = {
        'hidden_act': 'gelu', 'gate_attn_act_mode': 'sigmoid',
        'headwise_attn_output_gate': True, 'attention_bias': False, 'mlp_bias': False,
        'tie_word_embeddings': True, 'sliding_window': 512, 'rms_norm_eps': 1e-6,
        'max_position_embeddings': 1048576, 'bos_token_id': 0, 'eos_token_id': 1, 'pad_token_id': 2,
        'layer_types': ['sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention'] * 7,
    }
    if any(c.get(k) != v for k, v in required.items()):
        raise ValueError('Checkpoint architecture differs from Spark-X2.5-1.7B')
    for kind, theta, factor in [('sliding_attention', 10000, 1.0), ('full_attention', 5000000, 0.25)]:
        rope = c['rope_parameters'][kind]
        if rope['rope_theta'] != theta or rope['partial_rotary_factor'] != factor or rope.get('rope_type', 'default') != 'default':
            raise ValueError('Unsupported Spark RoPE configuration')
    h, f = c['hidden_size'], c['intermediate_size']
    shapes = {'model.embedding.weight': (c['vocab_size'], h), 'model.norm.weight': (h,)}
    for i in range(c['num_hidden_layers']):
        prefix = f'model.layers.{i}.'
        for name, shape in {
            'input_layernorm.weight': (h,), 'post_attention_layernorm.weight': (h,),
            'self_attn.q_k_v_proj.weight': (3072, h), 'self_attn.g_proj.weight': (8, h),
            'self_attn.out_proj.weight': (h, h), 'mlp.gate_proj.weight': (f, h),
            'mlp.up_proj.weight': (f, h), 'mlp.down_proj.weight': (h, f),
        }.items():
            shapes[prefix+name] = shape
    return shapes


def convert(checkpoint, output, quantized=False):
    config = json.loads((checkpoint/'config.json').read_text())
    shapes = expected_shapes(config)
    index = json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    if set(index) != set(shapes):
        raise ValueError('Checkpoint tensor inventory differs from Spark 1.7B')
    for shard in set(index.values()):
        if Path(shard).name != shard:
            raise ValueError('Invalid shard path')
        with safe_open(checkpoint/shard, framework='pt') as f:
            if set(f.keys()) != {n for n, s in index.items() if s == shard}:
                raise ValueError('Shard/index inventory mismatch')
            for name in f.keys():
                t = f.get_slice(name)
                if tuple(t.get_shape()) != shapes[name] or t.get_dtype() != 'BF16':
                    raise ValueError(f'Unexpected shape/dtype: {name}')
    if quantized:
        # Use the repository's portable KAI packer through the normal Python API.
        from pymllm.mobile.convertor.model_file_v2 import ModelFileV2
        from pymllm.mobile.quantize.pipeline import BUILTIN_QUANTIZE_PIPELINE
        recipe = json.loads(Path(__file__).with_name('quant_cfg_1.7B_w4a32_kai.json').read_text())
        params = {}
        for shard in sorted(set(index.values())):
            with safe_open(checkpoint/shard, framework='pt') as f:
                for name in f.keys():
                    params[name] = f.get_tensor(name)
                    if not torch.isfinite(params[name]).all():
                        raise ValueError(f"Nonfinite checkpoint tensor: {name}")
        partial = output.with_name(output.name+'.partial')
        if output.exists() or partial.exists():
            raise FileExistsError(output)
        pipeline = BUILTIN_QUANTIZE_PIPELINE['w4a32_kai_pipeline']()
        count = pipeline.stream_quantize_params_size(recipe, params)
        writer = ModelFileV2(str(partial), 'Spark-X2.5-1.7B-w4a32', 'Streaming',
                             max_params_descriptor_buffer_num=count)
        pipeline.stream_quantize(recipe, params, writer=writer, cast_left_2_fp32=True, verbose=True)
        partial.rename(output)
        return
    names = sorted(shapes)+['lm_head.weight']
    # Packed descriptors match pymllm/mobile/convertor/model_file_v2.py.
    # Reserve only the exact descriptor count; never overwrite an existing file.
    partial = output.with_name(output.name+'.partial')
    if output.exists() or partial.exists():
        raise FileExistsError(output)
    with partial.open('xb') as out:
        out.write(struct.pack('<II512sIQ', 0x519A, 2, b'Spark-X2.5-1.7B-fp32', len(names), 532))
        out.write(bytes(352*len(names)))
        for i, name in enumerate(names):
            source = 'model.embedding.weight' if name == 'lm_head.weight' else name
            with safe_open(checkpoint/index[source], framework='pt') as f:
                tensor = f.get_tensor(source).float().contiguous()
            if not torch.isfinite(tensor).all():
                raise ValueError(f'Nonfinite checkpoint tensor: {source}')
            offset = out.tell()
            payload = tensor.numpy().tobytes()
            out.write(payload)
            end = out.tell()
            shape = list(tensor.shape)
            out.seek(532+352*i)
            out.write(struct.pack('<IIQQQ16i256s', i, 0, len(payload), offset, len(shape),
                                  *(shape+[0]*(16-len(shape))), name.encode()))
            out.seek(end)
            print(name, shape, flush=True)
        out.flush()
        os.fsync(out.fileno())
    partial.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quantized', action='store_true', help='Use the built pymllm KAI W4A32 converter')
    args = parser.parse_args()
    convert(args.checkpoint, args.output, args.quantized)
