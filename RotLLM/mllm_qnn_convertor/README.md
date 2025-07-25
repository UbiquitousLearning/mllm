# MLLM QNN Convertor

In `RotLLM`, use the following command to profile the activation scales of a specific model:

```bash
python mllm_qnn_convertor/get_distribution_wobias.py --config_file mllm_qnn_convertor/config/get_distribution/Qwen2-get-dis.json
```

The content of the config file should be structured as follows:

```json
{
    "model_type": "qwen2",
    "model_name": "path/to/your/model",
    "tokenizer_name": "path/to/your/tokenizer",
    "dataset_path": "path/to/your/dataset",
    "output_file": "mllm_qnn_convertor/scales/Qwen2-7B-Rot_pile_val_dis.json",
    "num_samples": 64,
    "model_config": {
        "online_rotation": true,
        "random_rotate": true,
        "save_rotation": "mllm_qnn_convertor/R/Qwen2-7B-R.bin"
    }
}
```

Currently, we support the following model types: `qwen2` and `qwen2-vl`.

The activation scale information will be saved to the `output_file`, and the randomly generated rotation matrix will be saved to the `save_rotation` file.

## Export QNN Model

Use the following command to export a QNN-compatible model:

```bash
python mllm_qnn_convertor/export_qnn_model.py --config_file mllm_qnn_convertor/config/export/Qwen2-export-qnn.json
```

The content of the config file should be structured as follows:

```json
{
    "model_type": "qwen2",
    "model_name": "path/to/your/model",
    "tokenizer_name": "path/to/your/tokenizer",
    "scale_file": "mllm_qnn_convertor/scales/Qwen2-7B-Rot_pile_val_dis.json",
    "output_model": "path/to/output/model.bin",
    "model_config": {
        "online_rotation": true,
        "R_path": "mllm_qnn_convertor/R/Qwen2-7B-R.bin"
    },
    "export_config": {
        "t01m_clip_threshold": 64,
        "quant_bias": false
    }
}
```

The configuration format is similar to the get_distribution config. This command will save a PyTorch state_dict model file. You can then use `mllm/tools/convertor/convertor.py` as usual to generate a QNN MLLM model.

## Export FP32 Rotated Model

You can also use the following command to export an FP32 rotated model that can be converted to an MLLM CPU model:

```bash
python mllm_qnn_convertor/export_rotate_model.py --config_file mllm_qnn_convertor/config/export/Qwen2-export-rotate.json
```

The content of the config file should be structured as follows:

```json
{
    "model_type": "qwen2",
    "model_name": "path/to/your/model",
    "tokenizer_name": "path/to/your/tokenizer",
    "output_model": "path/to/output/rotated_model.bin",
    "model_config": {
        "online_rotation": true,
        "R_path": "mllm_qnn_convertor/R/Qwen2-7B-R.bin"
    }
}
```

This will export a rotated model in FP32 format that maintains the full precision while applying the rotation transformations.

## Workflow Summary

1. **Profile activation scales**: Run `get_distribution_wobias.py` to collect activation statistics and generate rotation matrices
2. **Export quantized model**: Run `export_qnn_model.py` to create a quantized model for QNN deployment
3. **Export FP32 rotated model**: Run `export_rotate_model.py` to create an FP32 rotated model for CPU deployment
4. **Convert to MLLM format**: Use the standard MLLM convertor tools to generate the final deployment model
