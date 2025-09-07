import os


def compile():
    COMMAND = [
        "cmake",
        "-G",
        "Ninja",
        "-B",
        "build",
        "-Dmllm_DIR=../SDK/lib/cmake/",
        "-DCMAKE_BUILD_TYPE=Release",
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
