# 使用基础镜像为 Ubuntu 22.04
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y wget 
RUN wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc &&\
    echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy main" | \
    tee -a /etc/apt/sources.list && \
    apt-get update
# 安装必要的软件包和工具链
RUN apt-get update && apt-get install -y \
    cmake \
    gdb \
    git \
    clang-format clang-tidy clang-tools clang clangd 