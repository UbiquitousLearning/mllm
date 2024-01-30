import torch

from TestUtils import TestBase


class CPUSilu1(TestBase):
    def test(self):
        input0 = torch.randn(1, 1, 3, 4)
        silu = torch.nn.SiLU()
        output = silu(input0)
        self.test_done(True)


if __name__ == '__main__':
    [instance().test() for instance in TestBase.__subclasses__()]
