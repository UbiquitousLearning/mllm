import os
import pymllm as mllm


def compile():
    COMMAND = [
        "cmake",
        "-G",
        "Ninja",
        "-B",
        "build-android",
        "-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_PATH/build/cmake/android.toolchain.cmake",
        "-DANDROID_PLATFORM=android-28",
        "-DANDROID_ABI=arm64-v8a",
        # Using your own mllm installation, please replace the path with your own path
        "-Dmllm_DIR=../SDK-Android/lib/cmake/",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    print(" ".join(COMMAND))
    os.system(" ".join(COMMAND))
    COMMAND = [
        "cmake",
        "--build",
        "build-android",
    ]
    print(" ".join(COMMAND))
    os.system(" ".join(COMMAND))


if __name__ == "__main__":
    compile()
    adb = mllm.utils.ADBToolkit()
    print(adb.get_devices())
    adb.push_file(
        "./build-android/lazy_vlm_qwen2vl",
        "/data/local/tmp/mllm/bin/algorithms/",
    )
    adb.push_file(
        "./build-android/lazy_vlm_qwen2_5vl",
        "/data/local/tmp/mllm/bin/algorithms/",
    )
    adb.push_file(
        "./build-android/lazy_vlm_qwen2vl_fast",
        "/data/local/tmp/mllm/bin/algorithms/",
    )
    adb.push_file(
        "./build-android/lazy_vlm_qwen2_5vl_fast",
        "/data/local/tmp/mllm/bin/algorithms/",
    )
    exit(0)
    with adb.get_shell_context() as shell:
        shell.execute("cd /data/local/tmp/mllm/bin/algorithms/")
        shell.execute(
            "export LD_LIBRARY_PATH=/data/local/tmp/mllm/bin:$LD_LIBRARY_PATH"
        )
        # If Using QNN, uncomment the following line
        # shell.execute("export ADSP_LIBRARY_PATH=/data/local/tmp/mllm/lib64")
        res = shell.execute(
            './lazy_vlm -m /data/local/tmp/mllm/models/Qwen2.5-VL-3B-Instruct/w4a32.mllm -mv v2 -t /data/local/tmp/mllm/models/Qwen2.5-VL-3B-Instruct/tokenizer.json -c /data/local/tmp/mllm/models/Qwen2.5-VL-3B-Instruct/config_w4a32.json -p "Describe this image" -i /data/local/tmp/mllm/bin/gafei.jpeg -s normal'
        )
        print(res)
