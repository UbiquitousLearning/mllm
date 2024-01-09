# mllm android project

## How to build

```shell
export ANDROID_NDK=/path/to/ndk
cd bulid_64
sh ../build.sh
```

## How to run

```shell
sh ../device_exe.sh
```

# execute qnn model

```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/mllm/bin:/data/local/tmp/mllm/qnn/qnn-lib
export ADSP_LIBRARY_PATH=/data/local/tmp/mllm/qnn/qnn-lib