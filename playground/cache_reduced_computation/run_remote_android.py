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
        "./build-android/cache_reduced_computation",
        "/data/local/tmp/mllm/bin/playground/",
    )
