"""
变异Pool算子，input*=delta【3个pool都可以】
已验证，成功
"""

import copy
import torch
import torch.nn as nn
import numpy as np
from torch import tensor
from torch_mutation.rules_torch.constant import *


class TransLayer_rule12_AvgPool2d(nn.Module):
    def __init__(self, layer_pool):
        super(TransLayer_rule12_AvgPool2d, self).__init__()

        self.layer_pool = layer_pool
        self.delta = tensor(np.random.uniform(-DELTA, DELTA, 1)[0]).to(device)

    def forward(self, x):
        mut_x = x * self.delta
        return self.layer_pool(mut_x) / self.delta

class TransLayer_rule12_MaxPool2d(nn.Module):
    def __init__(self, layer_pool):
        super(TransLayer_rule12_MaxPool2d, self).__init__()

        self.layer_pool = layer_pool
        self.delta = tensor(np.random.uniform(-DELTA, DELTA, 1)[0]).to(device)

    def forward(self, x):
        mut_x = x * self.delta
        return self.layer_pool(mut_x) / self.delta

class TransLayer_rule12_AdaptiveAvgPool2d(nn.Module):
    def __init__(self, layer_pool):
        super(TransLayer_rule12_AdaptiveAvgPool2d, self).__init__()

        self.layer_pool = layer_pool
        self.delta = tensor(np.random.uniform(-DELTA, DELTA, 1)[0]).to(device)

    def forward(self, x):
        mut_x = x * self.delta
        return self.layer_pool(mut_x) / self.delta







"""
torch.manual_seed(0)
np.random.seed(0)


DELTA = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


original_pool = nn.AvgPool2d(kernel_size=2, stride=2).to(device)

trans_pool = TransLayer(original_pool).to(device)


input_tensor = torch.randn(1, 1, 4, 4).to(device)


original_output = original_pool(input_tensor)


trans_output = trans_pool(input_tensor)


print("原始池化层输出:")
print(original_output)

print("\n变异池化层输出:")
print(trans_output)


diff = torch.abs(original_output - trans_output).max().item()
print("\n最大输出差异:", diff)
"""