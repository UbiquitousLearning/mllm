import pymllm as torch
import pymllm.nn as nn


class FooNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_0 = nn.Linear(1024, 1024)
        self.linear_1 = nn.Linear(1024, 1024)
        self.linear_2 = nn.Linear(1024, 1024)

    def forward(self, x: torch.Tensor):
        return self.linear_0(self.linear_1(self.linear_2(x)))


def test_foo_net():
    net = FooNet()
    print(net)
    t = torch.zeros((1, 1024), torch.float16, torch.cpu)
    print(t)
    o = net(t)
    print(o)


if __name__ == "__main__":
    test_foo_net()
