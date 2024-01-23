import torch

from TestUtils import TestBase


class CPULinear1(TestBase):
    def test(self):
        input0 = torch.randn(1, 8, 128)
        linear = torch.nn.Linear(in_features=128, out_features=128, bias=True)
        output = linear(input0)
        CPULinear_weight = linear.weight
        # print(CPULinear_weight.shape)
        CPULinear_bias = linear.bias

        # print(CPULinear_bias.shape)
        self.test_done(True)


class CPULinear2(TestBase):
    def test(self):
        input0 = torch.randn(1, 2, 3)
        linear = torch.nn.Linear(in_features=3, out_features=4, bias=False)
        output = linear(input0)
        CPULinear_weight = linear.weight
        # print(CPULinear_weight.shape)
        self.test_done(True)


#
# class CPULinear3(TestBase):
#     def test(self):
#         input0 = torch.randn(2, 2, 3)
#         linear = torch.nn.Linear(in_features=3, out_features=4, bias=False)
#         output = linear(input0)
#         CPULinear_weight = linear.weight
#         print(CPULinear_weight.shape)
#         self.test_done(True)


if __name__ == '__main__':
    [instance().test() for instance in TestBase.__subclasses__()]
