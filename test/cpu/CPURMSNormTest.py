import torch

from TestUtils import TestBase


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        y = x.pow(2).mean(-1, keepdim=True) + self.eps
        # print("y1", y)
        y = torch.sqrt(y)
        # print("y2", y)
        y = 1 / y
        # print("y3", y)
        return x * y

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)

        return output * self.weight


class CPURMSNorm1(TestBase):
    def test(self):
        seed = 1234
        torch.manual_seed(seed)
        torch.set_printoptions(precision=7)
        bs, seq_len, embedding_dim = 1, 10, 32000
        input0 = torch.randn(bs, seq_len, embedding_dim).float() * 1e-5
        rms = RMSNorm(embedding_dim, )
        output = rms(input0)
        CPURMSNorm_weight = rms.weight
        # print(CPURMSNorm_weight)
        # print(output)
        self.test_done(True)


if __name__ == '__main__':
    CPURMSNorm1().test()
