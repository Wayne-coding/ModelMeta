"""
变异BatchNorm算子，input+=delta,running_mean+=delta
已验证，成功
"""

import mindspore
import mindspore.nn as nn
import numpy as np
from mindspore import Tensor, Parameter
import mindspore.ops as ops
import copy


DTYPE = "float32"  
DELTA = 10         

class TransLayer_rule11(nn.Cell):
    def __init__(self, layer_bn):
        super(TransLayer_rule11, self).__init__()
        self.delta = Tensor(np.random.uniform(-DELTA, DELTA, 1)[0], mindspore.float32)
        self.layer_bn = layer_bn

        if self.layer_bn.use_batch_statistics:
            self.layer_bn.gamma = layer_bn.gamma
            self.layer_bn.beta = layer_bn.beta

            
            if "moving_mean" in layer_bn.parameters_dict():
                old_para = layer_bn.parameters_dict()
                new_para = copy.deepcopy(old_para)
                new_para["moving_mean"] = new_para["moving_mean"] + self.delta
                for key, value in new_para.items():
                    setattr(self.layer_bn, key, Parameter(value, requires_grad=False))

    def construct(self, x):
        x = x + self.delta
        return self.layer_bn(x)

if __name__ == "__main__" and False:
    
    batch_norm = nn.BatchNorm2d(num_features=3, affine=True, use_batch_statistics=True)
    batch_norm.set_train()

    
    x = Tensor(np.random.randn(10, 3, 32, 32), mindspore.float32)

    
    bn_output = batch_norm(x).asnumpy()
    

    
    trans_layer = TransLayer_rule11(batch_norm)
    if hasattr(trans_layer.layer_bn, "moving_mean"):
        print(trans_layer.layer_bn.moving_mean)

    
    trans_output = trans_layer(x).asnumpy()

    

    
    
    

    
    

    
    assert np.allclose(bn_output, trans_output, atol=1e-5), "The outputs are not matching!"
    print("The outputs match successfully!")
