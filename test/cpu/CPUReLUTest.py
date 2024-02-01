import torch

from TestUtils import TestBase
class CPUReLU1(TestBase):
    def test(self):
        seed = 1234
        torch.manual_seed(seed)
        torch.set_printoptions(precision=7)
        bs, seq_len, embedding_dim = 1, 10, 32000
        input0 = torch.randn(bs, seq_len, embedding_dim).float() * 1e-5
        relu = torch.nn.ReLU()
        output = relu(input0)
        # print(output)
        self.test_done(True)


if __name__ == '__main__':
    CPUReLU1().test()