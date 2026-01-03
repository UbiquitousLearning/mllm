import argparse
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

    args = parser.parse_args()

    m = Qwen3Quantizer(args.model_path, mllm_qualcomm_max_length=args.max_length)
    m.calibrate(num_samples=args.num_samples, max_seq_length=args.max_length)
    m.infer(args.infer_text)


if __name__ == "__main__":
    main()
