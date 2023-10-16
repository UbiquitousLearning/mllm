import torch

from TestUtils import TestBase


class CPUEmbedding1(TestBase):
    def test(self):
        input0 = torch.randint(0, 180, (1, 1, 128,))
        embedding = torch.nn.Embedding(180, 128)
        self.saver.write_tensor(embedding.weight, "CPUEmbedding.weight")
        output = embedding(input0)
        input0 = input0.to(torch.float32)
        print(output.shape)
        self.test_done(True)


if __name__ == '__main__':
    CPUEmbedding1().test()