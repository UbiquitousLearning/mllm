#!/usr/bin/env python3
"""
打印 safetensors 模型文件的所有键名
用法: python print_safetensors_keys.py <文件路径或目录>
"""

import sys
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:
    print("请先安装 safetensors 库:")
    print("  pip install safetensors")
    sys.exit(1)


def print_keys_from_file(filepath: Path):
    """打印单个 safetensors 文件的所有键名"""
    print(f"\n{'='*60}")
    print(f"文件: {filepath.name}")
    print(f"{'='*60}")
    
    with safe_open(filepath, framework="pt") as f:
        keys = f.keys()
        print(f"共 {len(keys)} 个键:\n")
        for i, key in enumerate(keys, 1):
            # 获取张量的形状信息
            tensor = f.get_tensor(key)
            shape = tuple(tensor.shape)
            dtype = tensor.dtype
            print(f"{key}")


def main():
    if len(sys.argv) < 2:
        print("用法: python print_safetensors_keys.py <文件路径或目录>")
        print("示例:")
        print("  python print_safetensors_keys.py model.safetensors")
        print("  python print_safetensors_keys.py ./models/")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    if path.is_file():
        # 单个文件
        if path.suffix == ".safetensors":
            print_keys_from_file(path)
        else:
            print(f"错误: {path} 不是 .safetensors 文件")
            sys.exit(1)
    elif path.is_dir():
        # 目录，查找所有 safetensors 文件
        files = sorted(path.glob("*.safetensors"))
        if not files:
            print(f"错误: 在 {path} 中未找到 .safetensors 文件")
            sys.exit(1)
        
        print(f"找到 {len(files)} 个 safetensors 文件")
        
        total_keys = 0
        for filepath in files:
            print_keys_from_file(filepath)
            with safe_open(filepath, framework="pt") as f:
                total_keys += len(f.keys())
        
        print(f"\n{'='*60}")
        print(f"总计: {len(files)} 个文件, {total_keys} 个键")
    else:
        print(f"错误: {path} 不存在")
        sys.exit(1)


if __name__ == "__main__":
    main()
