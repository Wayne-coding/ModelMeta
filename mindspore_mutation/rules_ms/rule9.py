""
"""
变异BatchNorm算子，bias+=delta
已验证，成功
"""


import mindspore.nn as nn
import mindspore.ops as ops
import mindspore
from mindspore import Tensor, Parameter
import numpy as np
from mindspore.common.initializer import initializer
from mindspore import context
import copy


DTYPE = "float32"  
DELTA = 10         




class TransLayerRule9(nn.Cell):
    def __init__(self, layer_bn):
        super(TransLayerRule9, self).__init__()

        self.layer_bn = copy.deepcopy(layer_bn)

        
        delta = np.random.uniform(-DELTA, DELTA, 1).astype(DTYPE)
        self.delta = Parameter(Tensor(delta, mindspore.float32), name="delta")
        self.layer_bn.beta += self.delta

        if self.layer_bn.use_batch_statistics:
            self.layer_bn.moving_mean = Parameter(self.layer_bn.moving_mean.clone(), name="moving_mean")
            self.layer_bn.moving_variance = Parameter(self.layer_bn.moving_variance.clone(), name="moving_variance")

    def construct(self, x):
        return self.layer_bn(x) - self.delta

if __name__ == "__main__" and False:
    
    batch_norm = nn.BatchNorm2d(10, use_batch_statistics=True)

    
    x = Tensor(np.random.randn(5, 10, 32, 32), mindspore.float32)

    
    bn_output = batch_norm(x)
    

    
    trans_layer = TransLayerRule9(batch_norm)

    
    trans_output = trans_layer(x)

    
    

    
    
    

    
    

    
    dis = ops.ReduceSum()(ops.Abs()(bn_output - trans_output))
    print("\nMaximum difference between BatchNorm and TransLayer outputs:")
    print(dis)