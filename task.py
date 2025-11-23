import os
import re
import sys
import time
import json
import yaml
import shutil
import logging
import platform
import argparse
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    format="%(asctime)s, [%(levelname)s]: %(message)s", level=logging.INFO
)

parser = argparse.ArgumentParser(description="Mllm task runner")
parser.add_argument("task_file", type=str, help="Path to task file")
args = parser.parse_args()
task_config = yaml.safe_load(Path(args.task_file).read_text())

PROJECT_ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__)))


def throw_error_if_failed(is_ok):
    if is_ok != 0:
        raise RuntimeError("FAILED!")


def wildcard_to_regex(pattern):
    escaped = re.escape(pattern)
    processed = escaped.replace(r"\*", ".*").replace(r"\?", ".")
    return f"^{processed}$"


def filter_files(directory, patterns, ignore_dirs=None, r=True, case_sensitive=True):
    root_dir = os.path.abspath(directory)

    ignore_abs = set()
    if ignore_dirs:
        for path in ignore_dirs:
            abs_path = os.path.normpath(os.path.join(root_dir, path))
            ignore_abs.add(abs_path.lower() if not case_sensitive else abs_path)

    flags = 0 if case_sensitive else re.IGNORECASE
    if isinstance(patterns, str):
        patterns = [patterns]
    regexes = [re.compile(wildcard_to_regex(p), flags) for p in patterns]

    matched = []

    if r:
        for root, dirs, files in os.walk(root_dir):
            current_path = os.path.normpath(root)
            compare_path = current_path.lower() if not case_sensitive else current_path

            if any(
                compare_path.startswith(ignored + os.sep) or compare_path == ignored
                for ignored in ignore_abs
            ):
                dirs[:] = []
                continue

            dirs[:] = [
                d
                for d in dirs
                if os.path.normpath(os.path.join(current_path, d)) not in ignore_abs
            ]

            for file in files:
                if any(rx.match(file) for rx in regexes):
                    matched.append(os.path.join(root, file))

    else:
        for entry in os.listdir(root_dir):
            full_path = os.path.join(root_dir, entry)
            if not os.path.isfile(full_path):
                continue

            file_path = os.path.normpath(full_path)
            compare_path = file_path.lower() if not case_sensitive else file_path
            if any(compare_path.startswith(ignored + os.sep) for ignored in ignore_abs):
                continue

            if any(rx.match(entry) for rx in regexes):
                matched.append(full_path)

    return matched


class Task:
    def __init__(self, config: Dict):
        self.config: Dict = config

    def make_command_str(self, commands: List) -> str:
        return " ".join(commands)

    def run(self):
        pass


class CMakeConfigTask(Task):
    def __init__(self, config):
        super().__init__(config)
        self.CMAKE_COMMAND = [
            "cmake",
            "-S",
            PROJECT_ROOT_PATH.as_posix(),
        ]

    def run(self):
        logging.info("CMake Config Task Start...")

        cmake_cfg_path = self.config.get("cmake_cfg_path", "build")
        self.CMAKE_COMMAND.extend(
            [
                "-G",
                "Ninja",
                "-B",
                os.path.join(PROJECT_ROOT_PATH, cmake_cfg_path),
            ]
        )

        cmake_build_type = self.config.get("cmake_build_type", "Release")
        self.CMAKE_COMMAND.extend(["-DCMAKE_BUILD_TYPE=" + cmake_build_type])

        cmake_toolchain_file = self.config.get("cmake_toolchain_file", None)
        if cmake_toolchain_file:
            self.CMAKE_COMMAND.extend(
                [
                    "-DCMAKE_TOOLCHAIN_FILE=" + cmake_toolchain_file,
                ]
            )

        cmake_extra_args = self.config.get("cmake_extra_args", None)
        if cmake_extra_args:
            self.CMAKE_COMMAND.extend(cmake_extra_args)

        commands = self.make_command_str(self.CMAKE_COMMAND)
        logging.info(f"{commands}")
        throw_error_if_failed(os.system(commands))

        logging.warning(
            f'If you are using vscode to develop. Pls set `"clangd.arguments": ["--compile-commands-dir={os.path.join(PROJECT_ROOT_PATH, cmake_cfg_path)}"]`'
        )

        show_ninja_targets = self.config.get("show_ninja_targets", False)
        if show_ninja_targets:
            logging.info("Finding targets in Ninja Builder:")
            throw_error_if_failed(
                os.system(
                    self.make_command_str(
                        [
                            "ninja",
                            "-C",
                            os.path.join(PROJECT_ROOT_PATH, cmake_cfg_path),
                            "-t",
                            "targets",
                        ]
                    )
                )
            )


class CMakeFormatTask(Task):
    def __init__(self, config):
        super().__init__(config)

    def run(self):
        logging.info("CMake Format Task Start...")

        ignore_path = self.config.get("ignore_path", [])
        cmake_files = filter_files(
            PROJECT_ROOT_PATH, ["*.cmake", "CMakeLists.txt"], ignore_path
        )
        for file in cmake_files:
            logging.info(f"cmake-format {file} -o {file}")
            throw_error_if_failed(os.system(f"cmake-format {file} -o {file}"))


class CMakeBuildTask(Task):
    def __init__(self, config):
        super().__init__(config)
        self.CMAKE_COMMAND = [
            "cmake",
            "--build",
            os.path.join(PROJECT_ROOT_PATH, self.config.get("cmake_cfg_path", "build")),
        ]

    def run(self):
        logging.info("CMake build Task Start...")
        targets = self.config.get("targets", None)
        if targets:
            for target in targets:
                sub_command = self.make_command_str(
                    self.CMAKE_COMMAND.extend(["--target", target])
                )
                logging.info(sub_command)
                throw_error_if_failed(os.system(sub_command))
        else:
            sub_command = self.make_command_str(self.CMAKE_COMMAND)
            logging.info(sub_command)
            throw_error_if_failed(os.system(sub_command))


class CMakeInstallTask(Task):
    def __init__(self, config):
        super().__init__(config)
        self.CMAKE_COMMAND = [
            "cmake",
            "--install",
            os.path.join(PROJECT_ROOT_PATH, self.config.get("cmake_cfg_path", "build")),
        ]

    def run(self):
        logging.info("Ninja install Task Start...")
        sub_command = self.make_command_str(self.CMAKE_COMMAND)
        logging.info(sub_command)
        throw_error_if_failed(os.system(sub_command))


class AdbPushTask(Task):
    def __init__(self, config):
        super().__init__(config)

    def run(self):
        logging.info("ADB push Task Start...")
        files = self.config["files"]
        push_path = self.config["to_path"]
        for file in files:
            command = [
                "adb",
                "push",
                file,
                push_path,
            ]
            logging.info(self.make_command_str(command))
            throw_error_if_failed(os.system(self.make_command_str(command)))


class ArmKernelBenchmarkTask(Task):
    def __init__(self, config):
        super().__init__(config)
        self.export_ld_path = (
            "export LD_LIBRARY_PATH=/data/local/tmp/mllm/bin:$LD_LIBRARY_PATH && "
        )

    def run(self):
        logging.info("Arm Kernel Benchmark Task Start...")
        benchmark_targets = self.config["benchmark_targets"]
        for target in benchmark_targets:
            file_name = os.path.basename(target)

            subcommand = [
                self.export_ld_path,
                target,
                "--benchmark_out_format=json",
                f"--benchmark_out={os.path.join(self.config['remote_results_path'], file_name + '.json')}",
            ]
            command = [
                "adb",
                "shell",
                f"'{self.make_command_str(subcommand)}'",
            ]
            logging.info(self.make_command_str(command))
            throw_error_if_failed(os.system(self.make_command_str(command)))
            logging.info("Waiting 16 seconds for device to calm down...")
            time.sleep(16)

        if not os.path.exists(PROJECT_ROOT_PATH / Path("temp")):
            os.mkdir(PROJECT_ROOT_PATH / Path("temp"))

        for target in benchmark_targets:
            file_name = os.path.basename(target)

            command = [
                "adb",
                "pull",
                f"{os.path.join(self.config['remote_results_path'], file_name + '.json')}",
                (PROJECT_ROOT_PATH / Path("temp")).as_posix(),
            ]
            logging.info(self.make_command_str(command))
            throw_error_if_failed(os.system(self.make_command_str(command)))

            md_headers = [
                "Name",
                "Run Name",
                "Run Type",
                "Iterations",
                "Real Time",
                "CPU Time",
                "Time Unit",
            ]
            with open(
                os.path.join(PROJECT_ROOT_PATH / Path("temp"), file_name + ".json"), "r"
            ) as file:
                data = json.load(file)
            rows = []
            for item in data["benchmarks"]:
                rows.append(
                    [
                        item["name"],
                        item["run_name"],
                        item["run_type"],
                        item["iterations"],
                        item["real_time"],
                        item["cpu_time"],
                        item["time_unit"],
                    ]
                )
            markdown = "| " + " | ".join(md_headers) + " |\n"
            markdown += "| " + " | ".join(["---"] * len(md_headers)) + " |\n"
            for row in rows:
                markdown += "| " + " | ".join(map(str, row)) + " |\n"

            with open(
                os.path.join(
                    PROJECT_ROOT_PATH
                    / Path("docs")
                    / Path("ArmBackend")
                    / Path("Benchmark"),
                    file_name + ".md",
                ),
                "w",
            ) as file:
                file.write(f"# {file_name} Benchmark Results\n\n")
                file.writelines(
                    [
                        f"device: {self.config['device_name']}\n\n",
                        f"data: {data['context']['date']}\n\n",
                        f"executable: {data['context']['executable']}\n\n",
                        f"num_cpus: {data['context']['num_cpus']}\n\n",
                        f"mhz_per_cpu: {data['context']['mhz_per_cpu']}\n\n",
                        f"cpu_scaling_enabled: {data['context']['cpu_scaling_enabled']}\n\n",
                        f"library_version: {data['context']['library_version']}\n\n",
                        f"library_build_type: {data['context']['library_build_type']}\n\n",
                    ]
                )
                file.write(markdown)


class GenPybind11StubsTask(Task):
    def __init__(self, config):
        super().__init__(config)

    def run(self):
        COMMANDS = [
            "pybind11-stubgen",
            "pymllm._C",
        ]
        throw_error_if_failed(os.system(self.make_command_str(COMMANDS)))
        logging.info(self.make_command_str(COMMANDS))
        tmp_C_path = PROJECT_ROOT_PATH / "stubs" / "pymllm"
        _C_path = PROJECT_ROOT_PATH / "pymllm"
        self.copy_files(tmp_C_path, _C_path)

    def copy_files(self, src_dir, dst_dir):
        import shutil

        os.makedirs(dst_dir, exist_ok=True)
        for root, dirs, files in os.walk(src_dir):
            rel_path = os.path.relpath(root, src_dir)
            dest_path = os.path.join(dst_dir, rel_path)

            os.makedirs(dest_path, exist_ok=True)

            for file in files:
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_path, file)
                shutil.copy(src_file, dest_file)


class BuildDocTask(Task):
    def __init__(self, config):
        super().__init__(config)

    def run(self):
        enable_doxygen = self.config.get("enable_doxygen", False)

        command_parts = ["python -m sphinx"]

        if enable_doxygen:
            command_parts.insert(0, "MLLM_ENABLE_DOXYGEN=true")

        command_parts.extend(
            [
                (PROJECT_ROOT_PATH / "docs").as_posix(),
                (PROJECT_ROOT_PATH / "docs" / "build").as_posix(),
            ]
        )

        command = " ".join(command_parts)
        throw_error_if_failed(os.system(command))
        logging.info(command)
        logging.info(
            f"Run `cd {PROJECT_ROOT_PATH / 'docs' / 'build'} && python -m http.server` to view the change."
        )


class HexagonMakeTask(Task):
    def __init__(self, config):
        super().__init__(config)
        self.mllm_qnn_package_place = config.get("mllm_qnn_package_place", None)
        assert self.mllm_qnn_package_place is not None, (
            "mllm_qnn_package_place must be set in config"
        )
        self.targets: List[str] = config.get("targets", None)
        assert self.targets is not None, "targets must be set in config"

    def run(self):
        # Get current working directory
        current_dir = os.getcwd()

        # Change to the mllm_qnn_package_place directory
        os.chdir(self.mllm_qnn_package_place)
        for target in self.targets:
            command = f"make {target}"
            logging.info(f"Running command: {command}")
            throw_error_if_failed(os.system(command))

        # Check if has htp_aarch64 in targets
        if "htp_aarch64" in self.targets:
            htp_aarch64_path = os.path.join(
                os.getcwd(), "build", "aarch64-android"
            )
            lib_obj = os.path.join(htp_aarch64_path, "libQnnLLaMAPackage.so")
            # Copy to libQnnLLaMAPackage_CPU.so
            if os.path.exists(lib_obj):
                print("found ", lib_obj)
                new_lib_obj = os.path.join(os.getcwd(), "build", "libQnnLLaMAPackage_CPU.so")
                logging.info(f"Copying {lib_obj} to {new_lib_obj}")
                shutil.copy(lib_obj, new_lib_obj)

        # Check if has htp_v75, htp_v68, htp_v69, htp_v73, htp_v79, htp_v81 in targets, and rename
        hexagon_versions = ["v75", "v68", "v69", "v73", "v79", "v81"]
        for version in hexagon_versions:
            if f"htp_{version}" in self.targets:
                htp_path = os.path.join(
                    os.getcwd(), "build", f"hexagon-{version}"
                )
                lib_obj = os.path.join(htp_path, "libQnnLLaMAPackage.so")
                # Copy to libQnnLLaMAPackage_HTP.so
                if os.path.exists(lib_obj):
                    new_lib_obj = os.path.join(os.getcwd(), "build", "libQnnLLaMAPackage_HTP.so")
                    logging.info(f"Copying {lib_obj} to {new_lib_obj}")
                    shutil.copy(lib_obj, new_lib_obj)

        # Change back to the original directory
        os.chdir(current_dir)


class MllmCliBuildTask(Task):
    def __init__(self, config):
        super().__init__(config)

    def run(self):
        logging.info("MLLM CLI Build Task Start...")

        build_dir = self.config.get("build_dir", "build")
        cli_root = self.config.get("cli_root", "mllm-cli")

        build_path = PROJECT_ROOT_PATH / build_dir
        build_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Build directory: {build_path}")

        go_build_cmd = [
            "go",
            "build",
            "-o",
            build_path.as_posix(),
        ]

        go_build_args = self.config.get("go_build_args", [])
        if go_build_args:
            go_build_cmd.extend(go_build_args)

        work_dir = PROJECT_ROOT_PATH / cli_root

        logging.info(f"Working directory: {work_dir}")
        logging.info(f"Command: {' '.join(go_build_cmd)}")

        original_cwd = os.getcwd()
        try:
            os.chdir(work_dir)
            throw_error_if_failed(os.system(self.make_command_str(go_build_cmd)))
        finally:
            os.chdir(original_cwd)

        sdk_lib_src = PROJECT_ROOT_PATH / cli_root / "mllm" / "SDK" / "lib"
        sdk_lib_dest = build_path / "lib"

        if sdk_lib_src.exists():
            logging.info(f"Copying SDK lib from {sdk_lib_src} to {sdk_lib_dest}")
            if sdk_lib_dest.exists():
                shutil.rmtree(sdk_lib_dest)
            shutil.copytree(sdk_lib_src, sdk_lib_dest)
        else:
            logging.warning(f"SDK lib directory not found: {sdk_lib_src}")

        sdk_include_src = PROJECT_ROOT_PATH / cli_root / "mllm" / "SDK" / "include"
        sdk_include_dest = build_path / "include"

        if sdk_include_src.exists():
            logging.info(
                f"Copying SDK include from {sdk_include_src} to {sdk_include_dest}"
            )
            if sdk_include_dest.exists():
                shutil.rmtree(sdk_include_dest)
            shutil.copytree(sdk_include_src, sdk_include_dest)
        else:
            logging.warning(f"SDK include directory not found: {sdk_include_src}")

        logging.info("MLLM CLI Build Task Completed")

class ShellCommandTask(Task):
    def __init__(self, config):
        super().__init__(config)

    def run(self):
        logging.info("Generic Shell Command Task Start (re-enabled)...")
        command_str = self.config.get("command", "")
        if not command_str:
            logging.error("No command provided in ShellCommandTask.")
            return

        throw_error_if_failed(os.system(command_str))

TASKS = {
    "CMakeConfigTask": CMakeConfigTask,
    "CMakeFormatTask": CMakeFormatTask,
    "CMakeBuildTask": CMakeBuildTask,
    "CMakeInstallTask": CMakeInstallTask,
    "AdbPushTask": AdbPushTask,
    "ArmKernelBenchmarkTask": ArmKernelBenchmarkTask,
    "BuildDocTask": BuildDocTask,
    "HexagonMakeTask": HexagonMakeTask,
    "MllmCliBuildTask": MllmCliBuildTask,
    "ShellCommandTask": ShellCommandTask, 
}


if __name__ == "__main__":
    for task_dict in task_config["Tasks"]:
        task_name = next(iter(task_dict))
        print(f"[Running task]: {task_name}")
        TASKS[task_name](task_dict[task_name]).run()
