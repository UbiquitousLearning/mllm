import torch

from TestUtils import TestBase


class CPUMul1(TestBase):
    def test(self):
        input0 = torch.randn(1, 2, 3, 4)
        input1 = torch.randn(1, 2, 3, 4)
        output = torch.mul(input0, input1)
        self.test_done(True)


if __name__ == '__main__':
    [instance().test() for instance in TestBase.__subclasses__()]
