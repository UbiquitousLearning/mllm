import torch

from TestUtils import TestBase


class CPULinear1(TestBase):
    def test(self):
        input0 = torch.randn(3, 1, 3, 3)
        output = torch.nn.Linear(in_features=3, out_features=4, bias=True)(input0)
        CPUlinear_weight = torch.nn.Linear(in_features=3, out_features=4, bias=True).weight
        print(CPUlinear_weight.shape)
        CPUlinear_bias = torch.nn.Linear(in_features=3, out_features=4, bias=True).bias
        print(CPUlinear_bias.shape)
        self.test_done(True)


if __name__ == '__main__':
    CPULinear1().test()
