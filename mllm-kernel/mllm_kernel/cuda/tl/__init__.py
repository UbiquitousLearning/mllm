try:
    import tilelang  # noqa: F401  # pyright: ignore[reportMissingImports]
except ImportError as exc:
    raise ImportError(
        "tilelang is required for mllm_kernel.cuda.tl. Please install tilelang first."
    ) from exc
