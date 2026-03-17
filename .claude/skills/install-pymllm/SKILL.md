---
name: install-pymllm
description: Install the pymllm Python package. Asks the user whether to do a full build (with CMake C++ compilation) or a fast install (Python-only, skip CMake). Use when the user asks to install, set up, or reinstall pymllm.
---

# Install pymllm

## Goal

Help the user install the `pymllm` package with the right configuration for their use case.

## Workflow

### Step 1: Ask the user which install mode they want

Use `AskUserQuestion` to present two options:

**Full Install (with C++ build)**
- Compiles the C++ mllm runtime and FFI extension via CMake
- Required if the user needs mobile inference, model conversion with FFI, or CPU/QNN backends
- Slower (several minutes depending on the machine)
- Command: `pip wheel -v -w dist . && pip install dist/*.whl --force-reinstall`

**Fast Install (Python-only, skip CMake)**
- Skips the entire CMake build step
- Only installs the pure Python package
- Recommended for users who only use CUDA backends (FlashInfer, TileLang) and do not need the C++ mllm runtime
- Much faster (seconds)
- Command: `SKBUILD_WHEEL_CMAKE=false pip install -e .`

### Step 2: Ask editable or non-editable

Use `AskUserQuestion` to ask:

- **Editable (`pip install -e .`)**: For active development. Python imports point to the source tree. Changes to `.py` files take effect immediately without reinstalling.
- **Non-editable (wheel)**: For stable usage. Installs a wheel into site-packages.

### Step 3: Execute the install

Based on user choices, run the appropriate command:

| Mode | Editable | Command |
|------|----------|---------|
| Full | Yes | `pip install -e -v .` |
| Full | No | `pip wheel -v -w dist . && pip install dist/*.whl --force-reinstall` |
| Fast | Yes | `SKBUILD_WHEEL_CMAKE=false pip install -e .` |
| Fast | No | `SKBUILD_WHEEL_CMAKE=false pip wheel -v -w dist . && pip install dist/*.whl --force-reinstall` |

### Step 4: Post-install for editable + full build

If the user chose **editable + full build**, the compiled `.so` files live in a build directory (e.g. `build/bin/`), not in the source tree. The Python code at `pymllm/__init__.py` looks for libraries at `pymllm/lib/MllmFFIExtension.so`. A symlink is needed to bridge this gap.

**Invoke the `/link-pymllm-lib` skill** to help the user set up the symlink.

### Step 5: Install optional CUDA dependencies

If the user chose fast install, suggest installing CUDA extras:

```bash
pip install pymllm[cuda]
```

This pulls in `tilelang`, `flashinfer-python`, and `pyzmq`.

## Important Notes

- The project root must contain `pyproject.toml` with `scikit-build-core` as the build backend.
- The `wheel.cmake = true` flag in `pyproject.toml` controls whether CMake runs. The env var `SKBUILD_WHEEL_CMAKE=false` overrides it at install time without modifying the file.
- For non-editable full builds, the `.so` files are bundled inside the wheel automatically — no symlink needed.
- For fast installs, `pymllm.is_mobile_available()` will return `False` since no C++ libraries are present. This is expected.
