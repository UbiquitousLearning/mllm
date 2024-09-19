import os
import sys
import argparse
from pathlib import Path
import logging
import shutil

PYTHON_EXE_PATH: Path = Path(sys.executable)
WORKING_DIR_ROOT: Path = Path.cwd()

parser = argparse.ArgumentParser(description='mllm workflow program')
parser.add_argument("-d", "--dev", required=False, action="store_true",
                    help="Using Development workflow for install python package")  # noqa: E501
parser.add_argument("-t", "--test", type=str, required=False,
                    help="Enable [python, C++] tests")
parser.add_argument("-c", "--compile-mllm-components",
                    required=False, action="store_true",
                    help="Compile all mllm components, but not install as python package")  # noqa: E501
args = parser.parse_args()


def copy_files(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    for root, dirs, files in os.walk(src_dir):
        rel_path = os.path.relpath(root, src_dir)
        dest_path = os.path.join(dst_dir, rel_path)

        os.makedirs(dest_path, exist_ok=True)

        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_path, file)
            shutil.copy(src_file, dest_file)


def run(res: bool):
    if res == 0:
        logging.info("Success")
    else:
        logging.error("Failed")
        exit(-1)


def pre_install_package() -> int:
    res = os.system(" ".join([
        f"{PYTHON_EXE_PATH}",
        "-m",
        "pip",
        "install",
        ".",
        "--no-build-isolation",
    ]))
    return res


def gen_stubs() -> int:
    tmp_path = WORKING_DIR_ROOT / "tmp"
    res = os.system(" ".join([
        "pybind11-stubgen",
        "mllm._C",
        "-o",
        "./tmp"
    ]))
    tmp_C_path = tmp_path / "mllm" / "_C"
    _C_path = WORKING_DIR_ROOT / "python" / "src" / "_C"

    # copy all files to EdgeInfer/_C
    copy_files(tmp_C_path, _C_path)
    return res


def install_package() -> int:
    return pre_install_package()


def dev():
    logging.info(f"find python bin: {PYTHON_EXE_PATH}")
    logging.info(
        f"pre-install edgeinfer package in directory: {WORKING_DIR_ROOT}")
    run(pre_install_package())
    logging.info("gen stubs(pyi files) for Pybindings")
    run(gen_stubs())
    logging.info("finalize install edgeinfer package")
    run(install_package())


def test():
    enabled_language = args.test.split(',')
    if "python" in enabled_language:
        logging.info("run pytest")
        run(os.system("pytest"))
    if "c++" in enabled_language:
        logging.info("run c++ google tests")


if __name__ == "__main__":
    if args.dev:
        dev()  # get dev version of edgeinfer installed.
    if args.test is not None:
        test()  # enable test
