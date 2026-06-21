pip install -r requirements.txt
pip install pre-commit
pre-commit install
pip install -r docs/requirements.txt

# Check GPU
nvidia-smi

git submodule update --init --recursive
git submodule update --remote --merge ./mllm/backends/cuda/vendors/cccl
git submodule update --remote --merge ./mllm/backends/cuda/vendors/cutlass
cd mllm/backends/cuda/vendors/
cd cccl
git checkout v3.0.2
cd ../cutlass
git checkout v4.1.0
