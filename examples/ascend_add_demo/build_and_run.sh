#!/bin/bash

# Ascend Add Demo 编译和运行脚本

set -e  # 遇到错误立即退出

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Ascend Add Demo 编译和运行脚本 ===${NC}\n"

# 检查环境变量
echo -e "${YELLOW}检查环境变量...${NC}"
if [ -z "$ASCEND_HOME_PATH" ]; then
    echo -e "${RED}错误: ASCEND_HOME_PATH 未设置${NC}"
    exit 1
fi
if [ -z "$ATB_HOME_PATH" ]; then
    echo -e "${RED}错误: ATB_HOME_PATH 未设置${NC}"
    exit 1
fi
echo -e "${GREEN}✓ ASCEND_HOME_PATH: $ASCEND_HOME_PATH${NC}"
echo -e "${GREEN}✓ ATB_HOME_PATH: $ATB_HOME_PATH${NC}\n"

# 获取项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build-ascend-demo"

echo -e "${YELLOW}项目根目录: $PROJECT_ROOT${NC}"
echo -e "${YELLOW}构建目录: $BUILD_DIR${NC}\n"

# 创建构建目录
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}创建构建目录...${NC}"
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

# 配置 CMake
echo -e "\n${YELLOW}配置 CMake...${NC}"
cmake "$PROJECT_ROOT" \
    -DMLLM_BUILD_ASCEND_BACKEND=ON \
    -DMLLM_ENABLE_EXAMPLE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# 编译
echo -e "\n${YELLOW}开始编译...${NC}"
make ascend_add_demo -j$(nproc)

# 检查编译结果
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ 编译成功！${NC}\n"
    
    # 运行
    echo -e "${YELLOW}运行 demo...${NC}\n"
    ./examples/ascend_add_demo/ascend_add_demo
    
    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}✓ Demo 运行成功！${NC}"
    else
        echo -e "\n${RED}✗ Demo 运行失败${NC}"
        exit 1
    fi
else
    echo -e "\n${RED}✗ 编译失败${NC}"
    exit 1
fi

