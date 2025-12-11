from pymllm.nn._layers import Softmax, RoPE


class QnnSoftmax(Softmax):
    def __init__(self):
        super().__init__()


class QnnRoPE(RoPE):
    def __init__(self):
        super().__init__()
