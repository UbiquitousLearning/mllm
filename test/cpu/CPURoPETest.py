import numpy as np
import torch

from TestUtils import TestBase


class RoPE(torch.nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, out):
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        return out


class CPURoPE1(TestBase):
    def test(self):
        input0 = torch.randn(180, 128)
        model = RoPE()
        output = model(input0)
        self.test_done(True)


if __name__ == '__main__':
    [instance().test() for instance in TestBase.__subclasses__()]
