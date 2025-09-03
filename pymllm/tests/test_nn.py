import pymllm as torch
import pymllm.nn as nn
import pymllm.nn.functional as F
from pymllm import ParameterFile


class FooNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_0 = nn.Linear(1024, 1024, False)
        self.linear_1 = nn.Linear(1024, 1024, False)
        self.linear_2 = nn.Linear(1024, 1024, False)

    def forward(self, x: torch.Tensor):
        return self.linear_0(self.linear_1(self.linear_2(x)))


def feed_fake_params(net: nn.Module) -> None:
    # Insert some random params
    param = ParameterFile()
    linear_0 = torch.random(
        (1024, 1024), dtype=torch.float32, device=torch.cpu
    ).set_name("linear_0.weight")
    linear_1 = torch.random(
        (1024, 1024), dtype=torch.float32, device=torch.cpu
    ).set_name("linear_1.weight")
    linear_2 = torch.random(
        (1024, 1024), dtype=torch.float32, device=torch.cpu
    ).set_name("linear_2.weight")
    param.push("linear_0.weight", linear_0)
    param.push("linear_1.weight", linear_1)
    param.push("linear_2.weight", linear_2)
    net.load(param)


def test_foo_net():
    net = FooNet()
    print(net)
    # Insert some random params
    param = ParameterFile()
    linear_0 = torch.random(
        (1024, 1024), dtype=torch.float32, device=torch.cpu
    ).set_name("linear_0.weight")
    linear_1 = torch.random(
        (1024, 1024), dtype=torch.float32, device=torch.cpu
    ).set_name("linear_1.weight")
    linear_2 = torch.random(
        (1024, 1024), dtype=torch.float32, device=torch.cpu
    ).set_name("linear_2.weight")
    param.push("linear_0.weight", linear_0)
    param.push("linear_1.weight", linear_1)
    param.push("linear_2.weight", linear_2)
    net.load(param)
    t = torch.random((1, 1024), dtype=torch.float32, device=torch.cpu)
    print(t)
    o = net(t)
    print(o)
    o = F.clip(o, min_val=-1.0, max_val=1.0)
    print(o)


if __name__ == "__main__":
    test_foo_net()
