# Building mllm - A Comprehensive Guide

This document provides detailed instructions on how to build the mllm project across different platforms and configurations.

## Prerequisites

Before building mllm, ensure you have the following tools installed:

- CMake 3.21 or higher
- C++20 compatible compiler (GCC, Clang, or MSVC)
- Python 3.8 or higher
- Git

## Platform-Specific Setup

### macOS (Apple Silicon)

For Apple Silicon Macs, use the following build command:

```bash
python task.py tasks/build_osx_apple_silicon.yaml
```

This configuration is optimized for Apple Silicon and uses GCD (Grand Central Dispatch) for threading.

### Linux (x86)

To build for Linux x86 systems:

```bash
python task.py tasks/build_x86.yaml
```

### Android

Building for Android requires the Android NDK. Set the `ANDROID_NDK_PATH` environment variable before building:

```bash
export ANDROID_NDK_PATH=/path/to/your/android/ndk
python task.py tasks/build_android.yaml
```

Additional Android-specific build targets:
- `tasks/build_android_opencl.yaml` - For OpenCL support
- `tasks/build_android_qnn.yaml` - For QNN support

### Docker

For consistent builds across platforms, you can use Docker:

```bash
cd ./docker
docker build -t mllm_arm -f Dockerfile.arm .
docker run -it --cap-add=SYS_ADMIN --network=host --cap-add=SYS_PTRACE --shm-size=4G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --name mllm_arm_dev mllm_arm bash
```

Note: The Docker image includes the Android NDK and requires accepting the NDK license terms.

## Backend Support

mllm supports multiple backends that can be enabled during build time:

- CPU (default, always available)
- OpenCL
- QNN (Qualcomm Neural Network SDK required)
- Metal (macOS only)

## Build Configuration Options

### Debug vs Release

The build system supports both debug and release configurations. Release builds are optimized for performance, while debug builds include additional checks and logging.

### Custom Build Targets

You can create custom build configurations by modifying or creating new YAML files in the `tasks` directory. These files specify build options, target platforms, and backend configurations.

## Python Bindings

To build Python bindings:

```bash
python task.py tasks/pymllm_install.yaml
```

For development, use the editable install:

```bash
python task.py tasks/pymllm_install_editable.yaml
```

## CLI Tool

To build the command-line interface tool:

```bash
python task.py tasks/build_osx_cli.yaml
```

Replace with the appropriate platform-specific task as needed.

## Troubleshooting

### Common Build Issues

1. **Missing Dependencies**: Ensure all required tools (CMake, compiler, Python) are installed and accessible from your PATH.

2. **Android NDK Issues**: Verify that ANDROID_NDK_PATH is correctly set and points to a valid NDK installation.

3. **CMake Version**: Make sure you're using CMake 3.21 or higher. Older versions may not support all required features.

4. **Compiler Compatibility**: Ensure your compiler supports C++20 features required by mllm.

### Platform-Specific Notes

- On Windows, you may need to enable symbol export with `-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON`
- On macOS, ensure you're using a compatible version of Xcode and command line tools
- Some Android devices may require specific build configurations for optimal performance
