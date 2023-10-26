import torch

from TestUtils import TestBase


class CPUMatmul1(TestBase):
    def test(self):
        input0 = torch.randn(512, 128)
        input1 = torch.randn(128, 256)
        output = torch.matmul(input0, input1)
        self.test_done(True)


class CPUMatmul2(TestBase):
    def test(self):
        input0 = torch.randn(2, 6)
        input1 = torch.randn(6, 10)
        output = torch.matmul(input0, input1)
        input1 = torch.transpose(input1, 0, 1)
        self.test_done(True)


class CPUMatmul3(TestBase):
    def test(self):
        input0 = torch.randn(9, 3)
        input1 = torch.randn(3, 1)
        output = torch.matmul(input0, input1)
        input1 = torch.transpose(input1, 0, 1)
        input0 = torch.transpose(input0, 0, 1)

        self.test_done(True)


if __name__ == '__main__':
    [instance().test() for instance in TestBase.__subclasses__()]
