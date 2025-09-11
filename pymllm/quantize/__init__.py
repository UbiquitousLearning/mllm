import subprocess
import os
import sys
from pathlib import Path

def run_quantizer(*args):
    """
    Run the mllm-quantizer executable with the provided arguments.
    
    Args:
        *args: Arguments to pass to the mllm-quantizer executable
        
    Returns:
        subprocess.CompletedProcess: Result of the subprocess execution
    """
    # 获取当前模块的路径
    current_dir = Path(__file__).parent
    # 构建可执行文件的预期路径
    quantizer_path = current_dir.parent / 'bin' / 'mllm-quantizer'
    
    # 检查可执行文件是否在包的bin目录中存在
    if not quantizer_path.exists():
        # 如果在预期位置找不到，尝试在系统PATH中查找
        quantizer_path = 'mllm-quantizer'
    
    # 构建完整的命令
    cmd = [str(quantizer_path)] + list(args)
    
    # 执行命令
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running mllm-quantizer: {e}", file=sys.stderr)
        print(f"stdout: {e.stdout}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        raise
    except FileNotFoundError:
        raise FileNotFoundError(
            "mllm-quantizer executable not found. "
            "Please make sure pymllm is properly installed with tools support."
        )

def quantize_model(input_model_path, output_model_path, quantization_type="q4_k_m"):
    """
    Quantize a model using the mllm-quantizer tool.
    
    Args:
        input_model_path (str): Path to the input model file
        output_model_path (str): Path where the quantized model will be saved
        quantization_type (str): Type of quantization to apply (default: "q4_k_m")
        
    Returns:
        subprocess.CompletedProcess: Result of the subprocess execution
    """
    return run_quantizer(
        "--input", input_model_path,
        "--output", output_model_path,
        "--type", quantization_type
    )