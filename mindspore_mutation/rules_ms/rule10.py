"""
变异BatchNorm算子，weight*=delta
已验证，成功
"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor, Parameter, context
from mindspore.common import dtype as mstype


device_target = "GPU"
context.set_context(mode=context.GRAPH_MODE, device_target=device_target)



DTYPE = "float32"  
DELTA = 10         


class TransLayerRule10(nn.Cell):
    def __init__(self, layer_bn):
        super(TransLayerRule10, self).__init__()
        self.layer_bn = layer_bn

        if self.layer_bn.use_batch_statistics:
            delta_value = np.random.uniform(1-DELTA, 1+DELTA, 1)[0].astype(np.float32)
            self.delta = Tensor(delta_value, mstype.float32)
            self.layer_bn.gamma = Parameter(self.layer_bn.gamma * self.delta, name='gamma')

        if self.layer_bn.use_batch_statistics:
            self.layer_bn.moving_mean = Parameter(self.layer_bn.moving_mean.copy(), requires_grad=False, name='moving_mean')
            self.layer_bn.moving_variance = Parameter(self.layer_bn.moving_variance.copy(), requires_grad=False, name='moving_variance')

    def construct(self, x):
        output = self.layer_bn(x)
        bias = ops.Reshape()(self.layer_bn.beta, (-1, 1, 1))
        return (output - bias) / self.delta + bias

if __name__ == "__main__" and False:
    
    batch_norm = nn.BatchNorm2d(3, use_batch_statistics=True)
    batch_norm.set_train()
    batch_norm.epsilon = 1e-3

    
    x = Tensor(np.random.randn(10, 3, 32, 32), mstype.float32)

    
    bn_output = batch_norm(x)

    
    trans_layer = TransLayerRule10(batch_norm)
    print("delta:")
    print(trans_layer.delta)
    print(trans_layer.layer_bn.moving_variance)

    
    trans_output = trans_layer(x)

    
    
    

    
    

    
    dis = ops.ReduceSum()(ops.Abs()(bn_output - trans_output))
    print("\nMaximum difference between BatchNorm and TransLayer outputs:")
    print(dis)