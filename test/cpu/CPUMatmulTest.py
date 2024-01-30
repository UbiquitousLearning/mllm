import torch

from TestUtils import TestBase


# class CPUMatmul1(TestBase):
#     def test(self):
#         input0 = torch.randn(8, 8)
#         input1 = torch.randn(8, 128)
#         output = torch.matmul(input0, input1)
#         self.test_done(True)


class CPUMatmul1(TestBase):
    def test(self):
        input0 = torch.randn(8, 8)
        input1 = torch.randn(8, 128)
        output = torch.matmul(input0, input1)
        input1 = torch.transpose(input1, 0, 1)
        self.test_done(True)


# class CPUMatmul3(TestBase):
#     def test(self):
#         input0 = torch.randn(8, 8)
#         input1 = torch.randn(8, 128)
#         output = torch.matmul(input0, input1)
#         # input1 = torch.transpose(input1, 0, 1)
#         input0 = torch.transpose(input0, 0, 1)
#
#         self.test_done(True)


if __name__ == '__main__':
    [instance().test() for instance in TestBase.__subclasses__()]
