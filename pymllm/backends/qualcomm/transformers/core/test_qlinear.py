import torch
import torch.nn as nn
from pymllm.backends.qualcomm.transformers.core.qlinear import QLinearLPBQ


def test_qlinear_lpbq():
    """
    Test QLinearLPBQ implementation against bf16 baseline.

    This test verifies that the double quantization implementation
    produces results close to the bf16 baseline when using appropriate
    quantization parameters.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Test parameters
    in_features = 256
    out_features = 128
    batch_size = 4
    seq_len = 16
    block_size = 64

    # Create input tensor (bf16 baseline)
    x_bf16 = torch.randn(batch_size, seq_len, in_features, dtype=torch.bfloat16)

    # Create reference linear layer (bf16)
    linear_bf16 = nn.Linear(in_features, out_features, bias=True, dtype=torch.bfloat16)
    # Copy weights and bias to ensure same values
    with torch.no_grad():
        linear_bf16.weight.copy_(
            torch.randn(out_features, in_features, dtype=torch.bfloat16)
        )
        linear_bf16.bias.copy_(torch.zeros(out_features, dtype=torch.bfloat16))

    # Get bf16 reference output
    with torch.no_grad():
        output_bf16 = linear_bf16(x_bf16)

    # Create QLinearLPBQ with same weights
    qlinear = QLinearLPBQ(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        block_size=block_size,
        already_quantized_weight=False,
        already_quantized_activation=False,
    )

    # Copy the same weights and bias
    with torch.no_grad():
        qlinear.weight.copy_(linear_bf16.weight.data)
        if qlinear.bias is not None:
            qlinear.bias.copy_(linear_bf16.bias.data)

    # Get quantized output
    with torch.no_grad():
        output_q = qlinear(x_bf16)
    output_q_bf16 = output_q

    # Calculate metrics
    mse = torch.mean((output_bf16 - output_q_bf16) ** 2)
    mae = torch.mean(torch.abs(output_bf16 - output_q_bf16))

    # Calculate relative error
    relative_error = torch.mean(
        torch.abs(output_bf16 - output_q_bf16) / (torch.abs(output_bf16) + 1e-8)
    )

    # Print results
    print("=== QLinearLPBQ Test Results ===")
    print(f"Input shape: {x_bf16.shape}")
    print(f"Output shape: {output_bf16.shape}")
    print(f"Block size: {block_size}")
    print("\nComparison with bf16 baseline:")
    print(f"MSE: {mse:.6e}")
    print(f"MAE: {mae:.6e}")
    print(f"Relative Error: {relative_error:.6e}")

    # Check if results are within acceptable tolerance
    # For double quantization, we expect some error but should be reasonable
    tolerance = 0.1  # 10% relative error tolerance

    if relative_error < tolerance:
        print(f"\n✓ TEST PASSED: Relative error {relative_error:.6e} < {tolerance}")
        return True
    else:
        print(f"\n✗ TEST FAILED: Relative error {relative_error:.6e} >= {tolerance}")
        return False
