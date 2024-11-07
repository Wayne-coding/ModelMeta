"""
变异BatchNorm算子，weight*=delta
已验证，成功
"""

import torch
import torch.nn as nn
import numpy as np
from torch import tensor
from torch_mutation.rules_torch.constant import *


class TransLayer_rule10(nn.Module):
    def __init__(self, layer_bn):
        super(TransLayer_rule10, self).__init__()
        self.layer_bn = layer_bn

        if self.layer_bn.affine:
            self.delta = torch.tensor(np.random.uniform(1-DELTA, 1+DELTA, 1)[0].astype(DTYPE), device=device)
            self.layer_bn.weight = nn.Parameter(self.layer_bn.weight * self.delta)

        if self.layer_bn.track_running_stats:
            self.layer_bn.register_buffer('running_mean', self.layer_bn.running_mean.clone())
            self.layer_bn.register_buffer('running_var', self.layer_bn.running_var.clone())

    def forward(self, x):
        return (self.layer_bn(x) - self.layer_bn.bias.reshape(-1, 1, 1)) / self.delta + self.layer_bn.bias.reshape(-1, 1, 1)


if __name__ == "__main__" and False:
    
    batch_norm = nn.BatchNorm2d(3, affine=True).to(device)
    batch_norm.train()
    batch_norm.eps = 1e-3

    
    x = torch.randn(10, 3, 32, 32).to(device)

    
    bn_output = batch_norm(x)

    
    trans_layer = TransLayer_rule10(batch_norm).to(device)
    print("delta:")
    print(trans_layer.delta)
    print(trans_layer.layer_bn.running_var)

    
    trans_output = trans_layer(x)

    
    print("\nBatchNorm output:")
    print(bn_output)

    print("\nTransLayer output:")
    print(trans_output)

    
    dis = torch.sum(torch.abs_(bn_output - trans_output))
    
    print("\nMaximum difference between BatchNorm and TransLayer outputs:")
    print(dis)