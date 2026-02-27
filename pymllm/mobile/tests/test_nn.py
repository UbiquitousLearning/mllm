import pymllm.mobile as mllm
from pymllm.mobile import nn


class FooModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.sf = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.sf(x)
        return x


if __name__ == "__main__":
    x = mllm.ones([6, 10])
    foo = FooModule()
    print(foo)
    print(foo(x))
