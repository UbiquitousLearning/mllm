---
name: link-pymllm-lib
description: Create or update the pymllm/lib symlink to point to a C++ build directory's bin/ folder. Required after editable installs with C++ builds so that Python can find the compiled .so libraries. Use when the user asks to link, fix, or set up pymllm native libraries.
---

# Link pymllm lib

## Goal

Create a symlink at `pymllm/lib` pointing to the correct build output directory so that an editable-installed pymllm can load the compiled C++ shared libraries (`MllmFFIExtension.so`, `libMllmRT.so`, etc.).

## Background

When pymllm is installed in editable mode (`pip install -e .`), Python imports from the source tree directly. The C++ libraries are compiled into `<build-dir>/bin/` by CMake, but pymllm looks for them at `pymllm/lib/`. A symlink bridges this gap:

```
pymllm/lib -> <project-root>/<build-dir>/bin
```

## Workflow

### Step 1: Detect available build directories

Scan the project root for directories matching the pattern `build*/bin/` that contain `MllmFFIExtension.so` (or `.dylib` on macOS). List all valid candidates.

Common build directories and their corresponding platforms:

| Build directory | Platform / Config | Typical build command |
|----------------|-------------------|----------------------|
| `build/bin` | X86 CPU only | `python task.py tasks/build_x86.yaml` |
| `build-x86-cuda/bin` | X86 + CUDA | `python task.py tasks/build_x86_cuda.yaml` |
| `build-qnn-aot/bin` | X86 + QNN AOT | `python task.py tasks/build_x86_qnn_aot.yaml` |
| `build-android-arm64-v8a-qnn/bin` | Android ARM + QNN | `python task.py tasks/build_android_qnn.yaml` |

### Step 2: Ask the user which build to link

Use `AskUserQuestion` to let the user pick from the detected build directories. Show each option with its path and the platform it corresponds to.

If no build directories with `.so` files are found, inform the user they need to build first:

```bash
pip install -r requirements.txt
python task.py tasks/build_x86.yaml  # or another build task
```

### Step 3: Check existing symlink

Before creating a new symlink, check if `pymllm/lib` already exists:

- If it's a symlink, show where it currently points and confirm replacement.
- If it's a real directory, warn the user and ask before removing it.
- If it doesn't exist, proceed directly.

### Step 4: Create the symlink

```bash
ln -sfn <project-root>/<build-dir>/bin <project-root>/pymllm/lib
```

Use `ln -sfn` to atomically replace any existing symlink.

### Step 5: Verify

After creating the symlink, verify by checking that the target `.so` file is accessible:

```bash
ls -la pymllm/lib/MllmFFIExtension.so
```

Then run a quick Python check:

```bash
python -c "import pymllm; print('mobile available:', pymllm.is_mobile_available())"
```

If `is_mobile_available()` returns `True`, the link is correct.

## Important Notes

- The symlink target must be an **absolute path** for reliability.
- On macOS, the library extension is `.dylib` instead of `.so`.
- Android build directories (e.g., `build-android-arm64-v8a-qnn/bin`) contain ARM binaries that cannot run on x86 hosts. Warn the user if they select one of these on a non-ARM machine.
- If the user has multiple build directories, they can re-run this skill anytime to switch which build pymllm uses.
