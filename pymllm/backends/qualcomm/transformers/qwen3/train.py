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
    m.infer(args.infer_text)

    # !!!
    # Things below is for deploy. We will turn all fp32 weights and some buffers(rope) to quantized dtype.
    # !!!
    # This line maybe error. we need use quantized weight!!! not embed_tokens.weight!!!
    # m.model.lm_head.weight = torch.nn.Parameter(
    #     m.model.model.embed_tokens.weight.clone()
    # )
    if "1.7B" in args.model_path:
        raise ValueError(
            "1.7B model is not supported for now due to tied embedding weights is not supported."
        )
    m.convert()

    os.makedirs(args.output_dir, exist_ok=True)
    model_save_path = os.path.join(args.output_dir, "model.safetensors")
    save_model(m.model, model_save_path)


if __name__ == "__main__":
    main()
