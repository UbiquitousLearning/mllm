# Ascend Add Op Demo

这是一个简单的 demo，用于测试 Ascend 后端的 Add 算子实现。

## 功能

- 初始化 Ascend 后端和内存池
- 创建两个输入张量（shape: [2, 3]）
- 在 Ascend NPU 上执行 Add 操作
- 验证计算结果是否正确

## 编译和运行

### 方法 1: 使用自动化脚本（推荐）

```bash
cd /home/HwHiAiUser/mLLM/examples/ascend_add_demo
./build_and_run.sh
```

脚本会自动：
- 检查环境变量
- 配置 CMake
- 编译项目
- 运行 demo

### 方法 2: 手动编译

确保已经设置了必要的环境变量：
- `ASCEND_HOME_PATH`: Ascend SDK 路径（已设置: `/usr/local/Ascend/ascend-toolkit/latest`）
- `ATB_HOME_PATH`: ATB 库路径（已设置: `/usr/local/Ascend/nnal/nnal/atb/latest/atb/cxx_abi_0`）

在项目根目录下：

```bash
# 1. 创建构建目录
mkdir -p build-ascend-demo && cd build-ascend-demo

# 2. 配置 CMake
cmake .. \
    -DMLLM_BUILD_ASCEND_BACKEND=ON \
    -DMLLM_ENABLE_EXAMPLE=ON \
    -DCMAKE_BUILD_TYPE=Release

# 3. 编译
make ascend_add_demo -j$(nproc)

# 4. 运行
./examples/ascend_add_demo/ascend_add_demo
```

## 预期输出

```
=== Ascend Add Op Demo ===
1. Initializing Ascend backend...
   ✓ Ascend backend initialized

2. Creating input tensors...
   Input x shape: [2, 3]
   Input y shape: [2, 3]

3. Transferring tensors to Ascend device...
   ✓ Tensors transferred to Ascend

4. Executing Add operation on Ascend...
   ✓ Add operation completed

5. Transferring result back to CPU and verifying...
   Expected result: [11, 22, 33, 44, 55, 66]
   Actual result:   [11, 22, 33, 44, 55, 66]

✓ Test PASSED! All values match expected results.
```

## 注意事项

- 当前实现使用 float16 数据类型
- 需要 Ascend NPU 设备可用
- 确保已正确安装 Ascend SDK 和 ATB 库

