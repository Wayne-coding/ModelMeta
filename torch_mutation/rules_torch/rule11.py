"""
变异BatchNorm算子，input+=delta,running_mean+=delta
已验证，成功
"""

import torch
import torch.nn as nn
import numpy as np
from torch import tensor
from torch_mutation.rules_torch.constant import *
import copy

class TransLayer_rule11(nn.Module):
    def __init__(self, layer_bn):
        super(TransLayer_rule11, self).__init__()
        self.delta = tensor(np.random.uniform(-DELTA, DELTA, 1)[0].astype(DTYPE), device=device)
        self.layer_bn = layer_bn

        if self.layer_bn.affine:
            self.layer_bn.weight = layer_bn.weight
            self.layer_bn.bias = layer_bn.bias

        if self.layer_bn.track_running_stats:
            old_para = layer_bn.state_dict()
            new_para = copy.deepcopy(old_para)
            new_para["running_mean"] = new_para["running_mean"] + self.delta
            self.layer_bn.load_state_dict(new_para)

    def forward(self, x):
        x += self.delta
        return self.layer_bn(x)

"""

batch_norm = nn.BatchNorm2d(3, affine=True).to(device)
batch_norm.train()


x = torch.randn(10, 3, 32, 32).to(device)


bn_output = batch_norm(x)
print(batch_norm.running_mean)


trans_layer = TransLayer(batch_norm).to(device)
print(trans_layer.layer_bn.running_mean)



trans_output = trans_layer(x)


print(trans_layer.delta)



print("\nBatchNorm output:")
print(bn_output)

print("\nTransLayer output:")
print(trans_output)

dis = (bn_output - trans_output)
print(dis)"""