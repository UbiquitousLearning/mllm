import pytest
import torch

from mllm_kernel.cuda.jit import gptq_marlin_repack


CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _pack_rows(q_weight: torch.Tensor, num_bits: int) -> torch.Tensor:
    pack_factor = 32 // num_bits
    size_k, size_n = q_weight.shape
    packed = torch.zeros(
        (size_k // pack_factor, size_n),
        dtype=torch.int32,
        device=q_weight.device,
    )
    for i in range(pack_factor):
        packed.bitwise_or_(q_weight[i::pack_factor].int() << (num_bits * i))
    return packed


def _reference_gptq_marlin_repack_cpu(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1
    q_weight = torch.empty((size_k, size_n), dtype=torch.int32)
    for i in range(pack_factor):
        q_weight[i::pack_factor] = (
            (b_q_weight >> (num_bits * i)) & mask
        )[0 : q_weight[i::pack_factor].shape[0]]

    if perm.numel() == 0:
        perm = torch.arange(size_k, dtype=torch.int32)

    out = torch.empty(
        (size_k // 16, size_n * 16 // pack_factor),
        dtype=torch.int32,
    )
    n_tiles = size_n // 64
    tc_offsets = [0, 1, 8, 9]
    pack_idx = [0, 2, 4, 6, 1, 3, 5, 7]
    tile_size = 16 * 64 // pack_factor

    for k_tile in range(size_k // 16):
        for n_tile in range(n_tiles):
            tile = torch.empty((16, 64), dtype=torch.int32)
            for local_k in range(16):
                src_k = int(perm[k_tile * 16 + local_k].item())
                tile[local_k] = q_weight[src_k, n_tile * 64 : (n_tile + 1) * 64]

            flat = torch.empty(tile_size, dtype=torch.int32)
            for warp_id in range(4):
                for th_id in range(32):
                    tc_col = th_id // 4
                    tc_row = (th_id % 4) * 2
                    cur_n = warp_id * 16 + tc_col

                    vals = [int(tile[tc_row + off, cur_n].item()) for off in tc_offsets]
                    vals.extend(
                        int(tile[tc_row + off, cur_n + 8].item())
                        for off in tc_offsets
                    )

                    res = 0
                    for i, src_idx in enumerate(pack_idx):
                        res |= vals[src_idx] << (i * num_bits)
                    if res >= 1 << 31:
                        res -= 1 << 32
                    flat[th_id * 4 + warp_id] = res

            out[k_tile, n_tile * tile_size : (n_tile + 1) * tile_size] = flat

    return out


@CUDA_ONLY
@pytest.mark.parametrize(
    ("size_k", "size_n", "num_bits"),
    [(128, 64, 4), (256, 128, 4)],
)
def test_gptq_marlin_repack_outputs_shape(size_k: int, size_n: int, num_bits: int) -> None:
    pack_factor = 32 // num_bits
    b_q_weight = torch.empty(
        (size_k // pack_factor, size_n),
        dtype=torch.int32,
        device="cuda",
    )
    perm = torch.empty(0, dtype=torch.int32, device="cuda")

    out = gptq_marlin_repack(
        b_q_weight,
        perm,
        size_k=size_k,
        size_n=size_n,
        num_bits=num_bits,
    )

    assert out.dtype == torch.int32
    assert out.shape == (size_k // 16, size_n * 16 // pack_factor)


@CUDA_ONLY
@pytest.mark.parametrize(
    ("size_k", "size_n", "num_bits"),
    [(128, 64, 4), (256, 128, 4)],
)
def test_gptq_marlin_repack_accepts_explicit_perm(
    size_k: int,
    size_n: int,
    num_bits: int,
) -> None:
    pack_factor = 32 // num_bits
    b_q_weight = torch.empty(
        (size_k // pack_factor, size_n),
        dtype=torch.int32,
        device="cuda",
    )
    perm = torch.arange(size_k, dtype=torch.int32, device="cuda")

    out1 = gptq_marlin_repack(
        b_q_weight,
        perm,
        size_k=size_k,
        size_n=size_n,
        num_bits=num_bits,
    )
    out2 = gptq_marlin_repack(
        b_q_weight,
        perm,
        size_k=size_k,
        size_n=size_n,
        num_bits=num_bits,
    )

    assert torch.equal(out1, out2)


@CUDA_ONLY
@pytest.mark.parametrize(
    ("size_k", "size_n", "num_bits"),
    [(128, 64, 4), (256, 128, 4)],
)
def test_gptq_marlin_repack_identity_perm_matches_empty_perm(
    size_k: int,
    size_n: int,
    num_bits: int,
) -> None:
    pack_factor = 32 // num_bits
    b_q_weight = torch.empty(
        (size_k // pack_factor, size_n),
        dtype=torch.int32,
        device="cuda",
    )
    empty_perm = torch.empty(0, dtype=torch.int32, device="cuda")
    perm = torch.arange(size_k, dtype=torch.int32, device="cuda")

    baseline = gptq_marlin_repack(
        b_q_weight,
        empty_perm,
        size_k=size_k,
        size_n=size_n,
        num_bits=num_bits,
    )
    with_perm = gptq_marlin_repack(
        b_q_weight,
        perm,
        size_k=size_k,
        size_n=size_n,
        num_bits=num_bits,
    )

    assert torch.equal(baseline, with_perm)


@CUDA_ONLY
def test_gptq_marlin_repack_non_identity_perm_matches_reference() -> None:
    size_k, size_n, num_bits = 128, 64, 4
    torch.manual_seed(2026)
    q_weight = torch.randint(
        0,
        1 << num_bits,
        (size_k, size_n),
        dtype=torch.int32,
    )
    b_q_weight_cpu = _pack_rows(q_weight, num_bits)
    perm_cpu = torch.roll(torch.arange(size_k, dtype=torch.int32), 1)

    out = gptq_marlin_repack(
        b_q_weight_cpu.to(device="cuda"),
        perm_cpu.to(device="cuda"),
        size_k=size_k,
        size_n=size_n,
        num_bits=num_bits,
    )
    ref = _reference_gptq_marlin_repack_cpu(
        b_q_weight_cpu,
        perm_cpu,
        size_k=size_k,
        size_n=size_n,
        num_bits=num_bits,
    )

    assert torch.equal(out.cpu(), ref)


@CUDA_ONLY
@pytest.mark.parametrize(
    ("size_k", "size_n", "num_bits"),
    [(128, 64, 4), (256, 128, 4)],
)
def test_gptq_marlin_repack_handles_noncontiguous_perm(
    size_k: int,
    size_n: int,
    num_bits: int,
) -> None:
    pack_factor = 32 // num_bits
    b_q_weight = torch.empty(
        (size_k // pack_factor, size_n),
        dtype=torch.int32,
        device="cuda",
    )

    buffer = torch.empty(
        size_k * 2,
        dtype=torch.int32,
        device="cuda",
    )
    indices = torch.arange(size_k, dtype=torch.int32, device="cuda")
    buffer[::2] = indices
    buffer[1::2] = indices
    perm = buffer.as_strided((size_k,), (2,))
    assert not perm.is_contiguous()

    perm_contig = perm.contiguous()

    out_noncontig = gptq_marlin_repack(
        b_q_weight,
        perm,
        size_k=size_k,
        size_n=size_n,
        num_bits=num_bits,
    )
    out_contig = gptq_marlin_repack(
        b_q_weight,
        perm_contig,
        size_k=size_k,
        size_n=size_n,
        num_bits=num_bits,
    )

    assert torch.equal(out_noncontig, out_contig)


@CUDA_ONLY
@pytest.mark.parametrize(
    ("size_k", "size_n", "num_bits"),
    [(128, 64, 4), (256, 128, 4)],
)
@pytest.mark.parametrize(
    "perm_factory",
    [
        lambda size_k: torch.arange(size_k, dtype=torch.int32, device="cpu"),
        lambda size_k: torch.arange(size_k, dtype=torch.int64, device="cuda"),
        lambda size_k: torch.arange(size_k - 16, dtype=torch.int32, device="cuda"),
        lambda size_k: torch.full((size_k,), size_k, dtype=torch.int32, device="cuda"),
        lambda size_k: torch.full((size_k,), -1, dtype=torch.int32, device="cuda"),
    ],
    ids=[
        "device-mismatch",
        "dtype-mismatch",
        "length-mismatch",
        "out-of-range",
        "negative-index",
    ],
)
def test_gptq_marlin_repack_rejects_invalid_perm(
    size_k: int,
    size_n: int,
    num_bits: int,
    perm_factory,
) -> None:
    pack_factor = 32 // num_bits
    b_q_weight = torch.empty(
        (size_k // pack_factor, size_n),
        dtype=torch.int32,
        device="cuda",
    )
    perm = perm_factory(size_k)

    with pytest.raises(ValueError):
        gptq_marlin_repack(
            b_q_weight,
            perm,
            size_k=size_k,
            size_n=size_n,
            num_bits=num_bits,
        )
