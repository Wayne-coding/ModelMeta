"""
变异sigmoid算子，input.transpose
已验证，成功
"""

import mindspore
import mindspore.nn as nn
import numpy as np
from mindspore import Tensor
from mindspore import ops
from mindspore.common.initializer import Normal
from mindspore import context
from mindspore import Parameter


context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

DELTA = 10  
FORMAT = "NHWC"  

class TransLayerRule16(nn.Cell):
    def __init__(self, layer_sigmoid):
        super(TransLayerRule16, self).__init__()
        self.layer_sigmoid = layer_sigmoid
        self.delta = Parameter(Tensor(np.random.uniform(0, DELTA, 1)[0] * 0.00001, mindspore.float32), name="delta")  

    def construct(self, x):
        mut_x = x * (1 + self.delta)  
        if FORMAT == "NHWC":
            x = ops.Transpose()(x, (0, 3, 1, 2))
            x = self.layer_sigmoid(x)
            x = ops.Transpose()(x, (0, 2, 3, 1))
        else:
            x = self.layer_sigmoid(x)
        return x

if __name__ == "__main__" and False:
    
    layer_sigmoid = nn.Sigmoid()
    
    
    model = TransLayerRule16(layer_sigmoid)
    
    
    input_tensor = Tensor(np.random.rand(1, 3, 224, 224), mindspore.float32)
    
    
    original_output = model(input_tensor)
    
    
    mutated_tensor = input_tensor * 1.00001  
    
    
    mutated_output = model(mutated_tensor)
    
    
    difference = ops.Abs()(original_output - mutated_output).max()
    print("Max difference between original and mutated output:", difference)
    
    
    if difference > 1e-5:
        print("Warning: Difference exceeds acceptable threshold!")
    