import pytest
import torch
import torch.nn.functional as F

from mllm_kernel.cuda.jit import gptq_marlin_gemm, gptq_marlin_repack


CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _compute_scalar_type_id(
    exponent: int,
    mantissa: int,
    signed: bool,
    bias: int,
    finite_values_only: bool = False,
    nan_repr: int = 1,
) -> int:
    bit_offset = 0
    result = 0
    for value, width in [
        (exponent, 8),
        (mantissa, 8),
        (signed, 1),
        (bias, 32),
        (finite_values_only, 1),
        (nan_repr, 8),
    ]:
        result |= (int(value) & ((1 << width) - 1)) << bit_offset
        bit_offset += width
    return result


SCALAR_TYPE_UINT4B8_ID = _compute_scalar_type_id(0, 4, False, 8)


def _pack_checkpoint_weight(q_weight: torch.Tensor, num_bits: int) -> torch.Tensor:
    pack_factor = 32 // num_bits
    size_n, size_k = q_weight.shape
    packed = torch.zeros(
        (size_n, size_k // pack_factor),
        dtype=torch.int32,
        device=q_weight.device,
    )
    for i in range(pack_factor):
        packed.bitwise_or_(q_weight[:, i::pack_factor].int() << (num_bits * i))
    return packed


def _get_scale_perms() -> tuple[list[int], list[int]]:
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]]
        )
    return scale_perm, scale_perm_single


def _marlin_permute_scales(
    s: torch.Tensor, size_k: int, size_n: int, group_size: int
) -> torch.Tensor:
    scale_perm, scale_perm_single = _get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape((-1, size_n)).contiguous()


def _marlin_make_workspace(device: torch.device) -> torch.Tensor:
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(sms, dtype=torch.int, device=device, requires_grad=False)


@CUDA_ONLY
def test_gptq_marlin_gemm_matches_reference_for_uint4b8() -> None:
    torch.manual_seed(2026)
    device = torch.device("cuda")
    size_m = 13
    size_n = 64
    size_k = 128
    group_size = 32
    num_bits = 4

    q_weight = torch.randint(
        0,
        1 << num_bits,
        (size_n, size_k),
        dtype=torch.int32,
        device=device,
    )
    scales = (
        torch.rand(
            (size_n, size_k // group_size),
            dtype=torch.float16,
            device=device,
        )
        + 0.5
    )
    packed = _pack_checkpoint_weight(q_weight, num_bits=num_bits)
    empty = torch.empty(0, dtype=torch.int32, device=device)
    marlin_q = gptq_marlin_repack(
        packed.t().contiguous(),
        perm=empty,
        size_k=size_k,
        size_n=size_n,
        num_bits=num_bits,
    )
    marlin_s = _marlin_permute_scales(
        scales.t().contiguous(),
        size_k=size_k,
        size_n=size_n,
        group_size=group_size,
    )
    x = torch.randn((size_m, size_k), dtype=torch.float16, device=device)
    workspace = _marlin_make_workspace(device)

    out = gptq_marlin_gemm(
        a=x,
        c=None,
        b_q_weight=marlin_q,
        b_scales=marlin_s,
        global_scale=None,
        b_zeros=empty,
        g_idx=empty,
        perm=empty,
        workspace=workspace,
        b_q_type_id=SCALAR_TYPE_UINT4B8_ID,
        size_m=size_m,
        size_n=size_n,
        size_k=size_k,
        is_k_full=True,
        use_atomic_add=False,
        use_fp32_reduce=False,
        is_zp_float=False,
    )

    ref_weight = (q_weight.to(torch.float16) - 8) * scales.repeat_interleave(
        group_size, dim=1
    )
    ref_out = F.linear(x, ref_weight)
    rel_mean_err = torch.mean(torch.abs(out - ref_out)) / torch.mean(
        torch.abs(ref_out)
    )

    assert rel_mean_err < 0.04
