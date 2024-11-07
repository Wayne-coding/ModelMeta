'''
变异softmax算子，input+=delta
已验证，成功
'''

import copy
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor, context
from mindspore.common.initializer import initializer
from mindspore import dtype as mstype




class TransLayer_rule17(nn.Cell):
    def __init__(self, layer_softmax):
        super(TransLayer_rule17, self).__init__()

        self.layer_softmax = layer_softmax
        self.delta = Tensor(10, mstype.float32)

    def construct(self, x):
        mut_x = x + self.delta
        return self.layer_softmax(mut_x)

if __name__ == "__main__" and False:
    
    softmax = nn.Softmax(axis=1)

    
    trans_layer = TransLayer_rule17(softmax)

    
    x = initializer('normal', [5, 10], mstype.float32)  

    
    original_output = softmax(x)
    print("Original Softmax Output:")
    print(original_output)

    
    mutated_output = trans_layer(x)
    print("\nMutated Softmax Output:")
    print(mutated_output)

    
    print("Are the outputs equal?", np.allclose(original_output.asnumpy(), mutated_output.asnumpy(), atol=1e-5))

    
    print("\nDelta used for mutation:")
    print(trans_layer.delta)