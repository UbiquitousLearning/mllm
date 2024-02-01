import torch

from TestUtils import TestBase
class CPUConvolution3D1(TestBase):
    def test(self):
        input0 = torch.randn(3, 3, 2, 224, 224)
        conv3d = torch.nn.Conv3d(in_channels=3, out_channels=1280, kernel_size=(2, 14, 14), stride=(2, 14, 14), groups=1, bias=False)
        output = conv3d(input0)
        CPUConvolution3D_weight = conv3d.weight
        # print(input0.shape, input0)
        # print(CPUConvolution3D_weight)
        # print(output)
        # print(CPUConvolution3D_weight.shape)
        # CPUConvolution3D_bias= conv3d.bias
        # print(CPUConvolution3D_bias.shape)
        self.test_done(True)

# class CPUConvolution3D2(TestBase):
#     def test(self):
#         input0 = torch.randn(1, 3, 3, 224, 224)
#         conv3d = torch.nn.Conv3d(in_channels=3, out_channels=1280, kernel_size=(2, 14, 14), stride=(2, 14, 14), groups=1, bias=True)
#         output = conv3d(input0)
#         CPUConvolution3D_weight = conv3d.weight
#         print(CPUConvolution3D_weight.shape)
#         CPUConvolution3D_bias= conv3d.bias
#         print(CPUConvolution3D_bias.shape)
#         self.test_done(True)


if __name__ == '__main__':
    CPUConvolution3D1().test()
    # CPUConvolution3D2().test()
    # input0 = torch.randn(1, 2, 5, 5).float()
    # conv2d = torch.nn.Conv2d(in_channels=2, out_channels=6, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), groups=1, bias=True, padding='same')
    # output = conv2d(input0)
    # CPUConvolution3D_weight = conv2d.weight
    # print(CPUConvolution3D_weight.shape)
    # CPUConvolution3D_bias= conv2d.bias
    # print(CPUConvolution3D_bias.shape)