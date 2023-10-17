import torch

from TestUtils import TestBase


class CPUAdd1(TestBase):
    def test(self):
        input0 = torch.randn(2, 2)
        input1 = torch.randn(2, 2)
        output = torch.add(input0, input1)
        self.test_done(True)

# if __name__ == '__main__':
