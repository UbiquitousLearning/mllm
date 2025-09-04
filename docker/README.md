# Easy Setup with Docker and DevContainer for MLLM

To simplify developers' experience with MLLM, we provide ready-to-use Dockerfile and DevContainer configurations.

## 1. Using Dockerfile

```bash
git clone https://github.com/UbiquitousLearning/mllm.git
cd mllm/docker

# CPU
docker build -t mllm_arm -f Dockerfile.arm .
docker run -it --cap-add=SYS_ADMIN --network=host --cap-add=SYS_PTRACE --shm-size=4G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --name mllm_arm_dev mllm_arm bash

# NVIDIA GPU. Chose your CUDA version: Dockerfile.cuxxx
docker build -t mllm_cu124 -f Dockerfile.cu124 .
docker run -it --gpus all --cap-add=SYS_ADMIN --network=host --cap-add=SYS_PTRACE --shm-size=4G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --name mllm_cu124_dev mllm_cu124 bash
```

Important Notes:

1. Dockerfile.arm includes NDK downloads. By using this image, you agree to NDK's additional terms.
2. QNN SDK contains proprietary licensing terms. We don't bundle it in Dockerfile.qnn - please configure QNN SDK manually.


## 2. Using DevContainer

To set up with VS Code Dev Containers:

1. Install prerequisites:
    - Docker
    - VS Code
    - Dev Containers extension

2. Clone repository with submodules:

    ```shell
    git clone --recursive https://github.com/UbiquitousLearning/mllm.git
    ```

3. Open project in VS Code:

    ```shell
    code mllm
    ```

4. When prompted:

    "Folder contains a Dev Container configuration file. Reopen in container?"
    Click Reopen in Container

    (Alternatively: Press F1 → "Dev Containers: Reopen in Container")

The container will automatically build and launch with:

* All dependencies pre-installed
* Correct environment configuration
* Shared memory and security settings applied
