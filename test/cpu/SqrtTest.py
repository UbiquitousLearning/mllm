import torch

from TestUtils import TestBase


class Sqrt1(TestBase):
    def test(self):
        input = torch.randn(3, 4, 5)
        output = torch.sqrt(input)
        print(output)
        self.test_done(True)


if __name__ == '__main__':
    [instance().test() for instance in TestBase.__subclasses__()]
