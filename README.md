## Latest News

## Features

## Usage Examples

## Tested Devices

| Device | OS | CPU | GPU | NPU |
| :---: | :---: | :---: | :---: | :---: |
| PC-X86-w/oAVX512 | Ubuntu 22.04 | ✅ PASS | - | - |
| Xiaomi14-8Elite | Android 15 | ✅ PASS | - | ✅ PASS |
| OnePlus13-8Elite| Android 15 | PENDING | - | PENDING |
| MacMini-M4 | MacOS 15.5 | ✅ PASS | - | - |

## Installation

### Arm Android

```shell
pip install -r requirements.txt
python task.py tasks/build_android.yaml
```

If you need to compile QNN Backends, please install the QNN SDK first. For instructions on setting up the QNN environment, please refer to [QNN README](mllm/backends/qnn/README.md).

Once the environment is configured, you can compile MLLM using the following command.

```shell
pip install -r requirements.txt
python task.py tasks/build_android_qnn.yaml
```

### X86 PC

```shell
pip install -r requirements.txt
python task.py tasks/build_x86.yaml
```

### OSX(Apple Silicon)

```shell
pip install -r requirements-mini.txt
python task.py tasks/build_osx_apple_silicon.yaml
```

### Use Docker

The MLLM Team provides Dockerfile to help you get started quickly, and we recommend using Docker images. In the `./docker/` folder, we provide images for arm (cross-compile to arm, host: x86) and qnn (cross-compile to arm, host: x86). Both ARM and QNN images support compilation of X86 Backends.

```bash
git clone https://github.com/UbiquitousLearning/mllm.git
cd mllm/docker
docker build -t mllm_arm -f Dockerfile.arm .
docker run -it --cap-add=SYS_ADMIN --network=host --cap-add=SYS_PTRACE --shm-size=4G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --name mllm_arm_dev mllm_arm bash
```

Important Notes:

1. Dockerfile.arm includes NDK downloads. By using this image, you agree to NDK's additional terms.
2. QNN SDK contains proprietary licensing terms. We don't bundle it in Dockerfile.qnn - please configure QNN SDK manually.

The details of how to use Dockerfile can be found in [Easy Setup with Docker and DevContainer for MLLM](docker/README.md)

### OpenCL Backend

## Quick Starts

## Tools

## Join us & Contribute

## Acknowledgements
