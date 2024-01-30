import torch
from torch import nn


class ReLUSquaredActivation(nn.Module):
    """
    Applies the relu^2 activation introduced in https://arxiv.org/abs/2109.08668v2
    """

    def forward(self, input):
        relu_applied = nn.functional.relu(input)
        squared = torch.square(relu_applied)
        return squared


from TestUtils import TestBase


class CPUReLU21(TestBase):
    def test(self):
        seed = 1234
        torch.manual_seed(seed)
        torch.set_printoptions(precision=7)
        bs, seq_len, embedding_dim = 1, 10, 32000
        input0 = torch.randn(bs, seq_len, embedding_dim).float()
        relu = ReLUSquaredActivation()
        output = relu(input0)
        # print(output)
        self.test_done(True)
