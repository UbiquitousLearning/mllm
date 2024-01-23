import torch

from TestUtils import TestBase
class CPUMaxPool2D1(TestBase):
    def test(self):
        input0 = torch.randn(1, 2, 5, 5)
        maxpool2d = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1))
        output = maxpool2d(input0)
        self.test_done(True)

class CPUMaxPool2D2(TestBase):
    def test(self):
        input0 = torch.randn(1, 2, 5, 5)
        maxpool2d = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        output = maxpool2d(input0)
        self.test_done(True)


if __name__ == '__main__':
    CPUMaxPool2D1().test()
    CPUMaxPool2D2().test()
    # input0 = torch.randn(1, 2, 5, 5).float()
    # conv2d = torch.nn.Conv2d(in_channels=2, out_channels=6, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), groups=1, bias=True, padding='same')
    # output = conv2d(input0)
    # CPUConvolution2D_weight = conv2d.weight
    # print(CPUConvolution2D_weight.shape)
    # CPUConvolution2D_bias= conv2d.bias
    # print(CPUConvolution2D_bias.shape)