def generate_clangd() -> None:
    import logging
    import os
    import pathlib
    import shutil
    import subprocess

    from tvm_ffi.libinfo import find_dlpack_include_path, find_include_path

    from mllm_kernel.jit_utils import (
        MLLM_KERNEL_CPU_INCLUDE_DIR,
        MLLM_KERNEL_CUDA_INCLUDE_DIR,
        MLLM_KERNEL_INCLUDE_DIR,
    )

    def _find_cuda_path() -> pathlib.Path | None:
        candidates: list[pathlib.Path] = []

        # 1) Standard CUDA environment variables.
        for env_name in ("CUDA_PATH", "CUDA_HOME"):
            env_value = os.environ.get(env_name)
            if env_value:
                candidates.append(pathlib.Path(env_value))

        # 2) Infer from nvcc in PATH.
        nvcc = shutil.which("nvcc")
        if nvcc:
            nvcc_path = pathlib.Path(nvcc).resolve()
            # .../bin/nvcc -> CUDA root
            candidates.append(nvcc_path.parent.parent)

        # 3) Common Linux install locations.
        candidates.extend(
            [
                pathlib.Path("/usr/local/cuda"),
                pathlib.Path("/opt/cuda"),
            ]
        )

        for path in candidates:
            if (path / "include" / "cuda_runtime.h").exists():
                return path
        return None

    logger = logging.getLogger()
    logger.info("Generating .clangd file...")
    include_paths = [find_include_path(), find_dlpack_include_path()] + [
        MLLM_KERNEL_CPU_INCLUDE_DIR,
        MLLM_KERNEL_CUDA_INCLUDE_DIR,
        MLLM_KERNEL_INCLUDE_DIR,
    ]
    # Keep include order stable while removing duplicates.
    include_paths = list(dict.fromkeys(str(path) for path in include_paths))
    status = subprocess.run(
        args=["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,
        check=True,
    )
    compute_cap = status.stdout.decode("utf-8").strip().split("\n")[0]
    major, minor = compute_cap.split(".")
    base_flags = [
        "-std=c++20",
        "-fexceptions",
        "-Wall",
        "-Wextra",
        *[f"-isystem{path}" for path in include_paths],
    ]
    cuda_flags = [
        "-xcuda",
        f"--cuda-gpu-arch=sm_{major}{minor}",
    ]

    cuda_path = _find_cuda_path()
    if cuda_path is not None:
        cuda_flags.append(f"--cuda-path={cuda_path}")
        logger.info(f"Detected CUDA installation at: {cuda_path}")
    else:
        logger.warning("CUDA installation not found; skip --cuda-path")

    def render_flags(flags: list[str]) -> str:
        return "\n".join(f"    - {flag}" for flag in flags)

    clangd_content = f"""
CompileFlags:
  Add:
{render_flags(base_flags)}

---
If:
  PathMatch: .*\\\\.(cu|cuh)$
CompileFlags:
  Add:
{render_flags(cuda_flags)}
"""
    if os.path.exists(".clangd"):
        logger.warning(".clangd file already exists, nothing done.")
        logger.warning(f"suggested content: {clangd_content}")
    else:
        with open(".clangd", "w") as f:
            f.write(clangd_content)
        logger.info(".clangd file generated.")


def show_env() -> None:
    import os
    import platform
    import re
    import shutil
    import subprocess
    import sys

    try:
        import torch
    except Exception:  # pragma: no cover - optional import failure path
        torch = None

    def _run_command(args: list[str]) -> str | None:
        try:
            status = subprocess.run(
                args=args,
                capture_output=True,
                check=True,
                text=True,
            )
            return status.stdout.strip()
        except Exception:
            return None

    def _linux_cpu_model_name() -> str | None:
        cpuinfo_path = "/proc/cpuinfo"
        if not os.path.exists(cpuinfo_path):
            return None
        try:
            with open(cpuinfo_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        _, _, model_name = line.partition(":")
                        return model_name.strip()
        except Exception:
            return None
        return None

    def _linux_physical_cpu_cores() -> int | None:
        cpuinfo_path = "/proc/cpuinfo"
        if not os.path.exists(cpuinfo_path):
            return None
        physical_ids: set[str] = set()
        cores_by_socket: dict[str, int] = {}
        current_physical_id = "0"
        try:
            with open(cpuinfo_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    key, _, value = line.partition(":")
                    key = key.strip().lower()
                    value = value.strip()
                    if key == "physical id":
                        current_physical_id = value
                        physical_ids.add(value)
                    elif key == "cpu cores":
                        try:
                            cores_by_socket[current_physical_id] = int(value)
                        except ValueError:
                            pass
        except Exception:
            return None

        if not cores_by_socket:
            return None
        if physical_ids:
            return sum(cores_by_socket.get(pid, 0) for pid in physical_ids)
        return sum(cores_by_socket.values())

    def _parse_nvcc_cuda_version(raw: str | None) -> str | None:
        if not raw:
            return None
        # Typical nvcc output contains: "release 12.4, V12.4.131"
        match = re.search(r"release\s+(\d+\.\d+)", raw, flags=re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _parse_nvidia_smi_driver_cuda_version(
        raw: str | None,
    ) -> tuple[str | None, str | None]:
        if not raw:
            return (None, None)
        # Header example: "... Driver Version: 550.54.14   CUDA Version: 12.4 ..."
        driver_match = re.search(
            r"Driver Version:\s*([^\s]+)",
            raw,
            flags=re.IGNORECASE,
        )
        cuda_match = re.search(
            r"CUDA Version:\s*([^\s]+)",
            raw,
            flags=re.IGNORECASE,
        )
        return (
            driver_match.group(1) if driver_match else None,
            cuda_match.group(1) if cuda_match else None,
        )

    logical_cores = os.cpu_count()
    physical_cores = _linux_physical_cpu_cores()
    cpu_model_name = _linux_cpu_model_name() or platform.processor() or "unknown"

    print("=== mllm-kernel environment ===")
    print("[OS]")
    print(f"system: {platform.system()}")
    print(f"release: {platform.release()}")
    print(f"version: {platform.version()}")
    print(f"platform: {platform.platform()}")
    print(f"machine: {platform.machine()}")
    print()
    print("[Python]")
    print(f"version: {sys.version.split()[0]}")
    print(f"executable: {sys.executable}")
    print()
    print("[CPU]")
    print(f"model: {cpu_model_name}")
    print(f"logical_cores: {logical_cores}")
    print(
        f"physical_cores: {physical_cores if physical_cores is not None else 'unknown'}"
    )
    print()
    print("[CUDA]")

    cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    nvcc_path = shutil.which("nvcc")
    nvidia_smi_path = shutil.which("nvidia-smi")

    print(f"CUDA_PATH/CUDA_HOME: {cuda_path if cuda_path else 'not set'}")
    print(f"nvcc_path: {nvcc_path if nvcc_path else 'not found'}")
    print(f"nvidia_smi_path: {nvidia_smi_path if nvidia_smi_path else 'not found'}")

    nvcc_version_raw = _run_command(["nvcc", "--version"]) if nvcc_path else None
    nvcc_cuda_version = _parse_nvcc_cuda_version(nvcc_version_raw)
    print(f"nvcc_cuda_version: {nvcc_cuda_version if nvcc_cuda_version else 'unknown'}")

    nvidia_smi_raw = _run_command(["nvidia-smi"]) if nvidia_smi_path else None
    nvidia_driver_version, nvidia_cuda_version = _parse_nvidia_smi_driver_cuda_version(
        nvidia_smi_raw
    )
    print(
        "nvidia_smi_driver_version: "
        f"{nvidia_driver_version if nvidia_driver_version else 'unknown'}"
    )
    print(
        f"nvidia_smi_cuda_version: {nvidia_cuda_version if nvidia_cuda_version else 'unknown'}"
    )

    if torch is None:
        print("torch_installed: no")
        return

    print("torch_installed: yes")
    print(f"torch_version: {torch.__version__}")
    print(f"torch_cuda_build_version: {torch.version.cuda}")
    print(f"torch_cuda_available: {torch.cuda.is_available()}")
    print(f"torch_cuda_device_count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(idx)
            major, minor = torch.cuda.get_device_capability(idx)
            print(f"gpu[{idx}]: {device_name}, compute_capability=sm_{major}{minor}")


def show_config() -> None:
    import importlib
    import pathlib
    import re

    from mllm_kernel.jit_utils import get_jit_kernel_registry
    from mllm_kernel.jit_utils.compile import MLLM_KERNEL_CACHE_ROOT

    def _sanitize_cache_name(name: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._-")
        return sanitized or "jit_kernel"

    def _build_cache_name(
        device_prefix: str,
        func_name: str,
        template_args: tuple[str, ...],
    ) -> str:
        if not template_args:
            return f"{device_prefix}_{func_name}"
        return f"{device_prefix}_{func_name}_" + "_".join(template_args)

    def _resolve_cache_directory(
        info: dict[str, object],
        target_device: str,
    ) -> pathlib.Path:
        build_directory = info.get("build_directory")
        if isinstance(build_directory, str) and build_directory:
            return pathlib.Path(build_directory).expanduser()

        export_name = str(info.get("export_name", "jit_kernel"))
        template_args_raw = info.get("template_args")
        template_args: tuple[str, ...]
        if isinstance(template_args_raw, tuple):
            template_args = tuple(str(arg) for arg in template_args_raw)
        else:
            template_args = ()

        cache_name = _build_cache_name(target_device, export_name, template_args)
        return MLLM_KERNEL_CACHE_ROOT / _sanitize_cache_name(cache_name)

    def _has_cached_shared_object(cache_directory: pathlib.Path) -> bool:
        if not cache_directory.exists() or not cache_directory.is_dir():
            return False
        return any(
            path.is_file() and path.name.endswith(".so")
            for path in cache_directory.rglob("*.so")
        )

    def _format_kernel_name(info: dict[str, object]) -> str:
        export_name = str(info.get("export_name", "unknown_kernel"))
        template_args_raw = info.get("template_args")
        if isinstance(template_args_raw, tuple) and template_args_raw:
            template_args = ", ".join(str(arg) for arg in template_args_raw)
            return f"{export_name}<{template_args}>"
        return export_name

    def _colorize_cached(cached: str, *, is_cached: bool) -> str:
        green = "\033[32m"
        red = "\033[31m"
        reset = "\033[0m"
        color = green if is_cached else red
        return f"{color}{cached}{reset}"

    # Make sure built-in kernels are imported so their decorators register entries.
    for module_name in ("mllm_kernel.cpu.jit", "mllm_kernel.cuda.jit"):
        try:
            importlib.import_module(module_name)
        except Exception:
            pass

    show_env()
    print()
    print("=== registered jit kernels ===")

    registry = get_jit_kernel_registry()
    if not registry:
        print("device | kernel | cached")
        print("no registered kernels")
        return

    rows: list[tuple[str, str, str]] = []
    for info in registry.values():
        info_dict = dict(info)
        configured_device = str(info_dict.get("decorator_device", "unknown"))
        target_devices = (
            ("cpu", "cuda") if configured_device == "auto" else (configured_device,)
        )
        kernel_name = _format_kernel_name(info_dict)
        for target_device in target_devices:
            cache_directory = _resolve_cache_directory(info_dict, target_device)
            cached = "yes" if _has_cached_shared_object(cache_directory) else "no"
            rows.append((target_device, kernel_name, cached))

    rows.sort(key=lambda row: (row[0], row[1]))

    header = ("device", "kernel", "cached")
    device_width = max(len(header[0]), *(len(device) for device, _, _ in rows))
    kernel_width = max(len(header[1]), *(len(kernel) for _, kernel, _ in rows))
    cached_width = max(len(header[2]), *(len(cached) for _, _, cached in rows))

    print(
        f"{header[0]:<{device_width}} | {header[1]:<{kernel_width}} | {header[2]:<{cached_width}}"
    )
    print(f"{'-' * device_width}-+-{'-' * kernel_width}-+-{'-' * cached_width}")
    for device, kernel, cached in rows:
        cached_padded = f"{cached:<{cached_width}}"
        print(
            f"{device:<{device_width}} | {kernel:<{kernel_width}} | "
            f"{_colorize_cached(cached_padded, is_cached=(cached == 'yes'))}"
        )


def main() -> None:
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        prog="python -m mllm_kernel",
        description="mllm-kernel helper commands.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["show-clangd-recommend-config", "show-env", "show-config"],
        help=(
            "Run helper command. "
            "Use 'show-clangd-recommend-config' to generate .clangd, "
            "'show-env' to print runtime environment details, "
            "or 'show-config' to print environment and JIT cache status."
        ),
    )
    args = parser.parse_args()

    if args.command == "show-clangd-recommend-config":
        generate_clangd()
        return
    if args.command == "show-env":
        show_env()
        return
    if args.command == "show-config":
        show_config()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
