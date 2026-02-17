import os
import torch
import argparse
from safetensors.torch import save_model
from pymllm.backends.qualcomm.transformers.qwen3.runner import Qwen3Quantizer


def main():
    parser = argparse.ArgumentParser(description="Qwen3 Quantizer for Qualcomm backend")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen3-1.7B",
        help="Path to the Qwen3 model directory",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length for quantization",
    )
    parser.add_argument(
        "--num_samples", type=int, default=128, help="Number of samples for calibration"
    )
    parser.add_argument(
        "--infer_text",
        type=str,
        default="为什么伟大不能被计划",
        help="Text to run inference on",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the quantized model",
    )

    args = parser.parse_args()

    m = Qwen3Quantizer(args.model_path, mllm_qualcomm_max_length=args.max_length)

    # FIXME: Should disable or not.
    m.disable_fake_quant()
    m.calibrate(num_samples=args.num_samples, max_seq_length=args.max_length)
    m.enable_fake_quant()
    m.recompute_scale_zp()
    m.validate_concat_observer()
    m.infer(args.infer_text)
    m.convert()

    os.makedirs(args.output_dir, exist_ok=True)
    model_save_path = os.path.join(args.output_dir, "model.safetensors")
    save_model(m.model, model_save_path)


if __name__ == "__main__":
    main()
