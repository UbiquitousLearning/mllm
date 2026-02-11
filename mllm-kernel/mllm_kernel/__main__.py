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
        choices=["show-clangd-recommend-config"],
        help="Run helper command. Use 'show-clangd-recommend-config' to generate .clangd.",
    )
    args = parser.parse_args()

    if args.command == "show-clangd-recommend-config":
        generate_clangd()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
