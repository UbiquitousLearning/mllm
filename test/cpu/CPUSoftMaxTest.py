import torch

from TestUtils import TestBase


class CPUSoftMax1(TestBase):
    def test(self):
        input0 = torch.randn(2, 3, 4, 5, dtype=torch.float32)
        softmax = torch.nn.Softmax(dim=-1)
        output = softmax(input0)
        # print(output.shape, output.dtype)
        self.test_done(True)


if __name__ == '__main__':
    [instance().test() for instance in TestBase.__subclasses__()]
