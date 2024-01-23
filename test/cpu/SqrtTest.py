import torch

from TestUtils import TestBase


class Sqrt1(TestBase):
    def test(self):
        input = torch.randn(1, 2) * 1e-6
        output = torch.sqrt(input)
        # print(output)
        self.test_done(True)


if __name__ == '__main__':
    [instance().test() for instance in TestBase.__subclasses__()]
