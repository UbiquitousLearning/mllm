pip install -r requirements.txt
pip install pre-commit
pre-commit install
pip install -r docs/requirements.txt

# Check GPU
nvidia-smi

git submodule update --remote --merge ./mllm/backends/cuda/vendors/cccl
git submodule update --remote --merge ./mllm/backends/cuda/vendors/cutlass
