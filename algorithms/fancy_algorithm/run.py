import os
import pymllm as mllm


def compile():
    COMMAND = [
        "cmake",
        "-G",
        "Ninja",
        "-B",
        "build",
        "-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_PATH/build/cmake/android.toolchain.cmake",
        "-DANDROID_PLATFORM=android-28",
        "-DANDROID_ABI=arm64-v8a",
        # Using your own mllm installation, please replace the path with your own path
        "-Dmllm_DIR=/root/mllm-install-android-arm64-v8a/lib/cmake/",
    ]
    print(" ".join(COMMAND))
    os.system(" ".join(COMMAND))
    COMMAND = [
        "cmake",
        "--build",
        "build",
    ]
    print(" ".join(COMMAND))
    os.system(" ".join(COMMAND))


if __name__ == "__main__":
    compile()
    adb = mllm.utils.ADBToolkit()
    print(adb.get_devices())
    adb.push_file(
        "./build/fancy_algorithm",
        "/data/local/tmp/mllm/bin/algorithms/",
    )
    with adb.get_shell_context() as shell:
        shell.execute("cd /data/local/tmp/mllm/bin/algorithms/")
        shell.execute(
            "export LD_LIBRARY_PATH=/data/local/tmp/mllm/bin:$LD_LIBRARY_PATH"
        )
        # If Using QNN, uncomment the following line
        # shell.execute("export ADSP_LIBRARY_PATH=/data/local/tmp/mllm/lib64")
        res = shell.execute("./fancy_algorithm --help")
        print(res)
