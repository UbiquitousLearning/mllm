from pymllm.backends.qualcomm.transformers.qwen3.runner import Qwen3Quantizer

if __name__ == "__main__":
    m = Qwen3Quantizer()
    m.calibrate()
    m.infer("简述中国断代史")
