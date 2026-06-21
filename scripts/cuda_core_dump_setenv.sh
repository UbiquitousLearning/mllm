export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
export CUDA_COREDUMP_SHOW_PROGRESS=1
export CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory'

# Optional(For containers)
# export CUDA_COREDUMP_FILE="/persistent_dir/cuda_coredump_%h.%p.%t"