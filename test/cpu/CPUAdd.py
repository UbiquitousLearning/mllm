import torch

import TestUtils

input1 = torch.randn(2,2)
input2 = torch.randn(2,2)
print(input1)
print(input2)
#TODO: Save Tensor?
saver = TestUtils.TestSaver("CPUAdd1")
saver.write_tensor(input1, "input0")
saver.write_tensor(input2, "input1")
output = torch.add(input1, input2)
print(output)
saver.write_tensor(output, "output")
