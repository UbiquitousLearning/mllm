import torch

from TestUtils import TestBase
class CPULayerNorm1(TestBase):
    def test(self):
        bs, seq_len, embedding_dim = 1, 10, 32000
        input0 = torch.randn(bs, seq_len, embedding_dim).float()
        layer_norm = torch.nn.LayerNorm(embedding_dim, eps=1e-5)
        output = layer_norm(input0)
        CPULayerNorm_weight = layer_norm.weight
        CPULayerNorm_bias = layer_norm.bias
        # print(CPULayerNorm_weight)
        # print(CPULayerNorm_bias)
        # print(output)
        self.test_done(True)