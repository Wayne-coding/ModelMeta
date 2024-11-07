"""
变异tanh算子，input*(-1)
已验证，成功
"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor, context

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

class TransLayerRule18(nn.Cell):
    def __init__(self, layer_tanh):
        super(TransLayerRule18, self).__init__()
        self.layer_tanh = layer_tanh

    def construct(self, x):
        return -self.layer_tanh(-x)

if __name__ == "__main__" and False:
    
    tanh = nn.Tanh()

    
    trans_layer = TransLayerRule18(tanh)

    
    x = Tensor(np.random.randn(5, 1), mindspore.float32)

    
    original_output = tanh(x)
    print(original_output)

    
    mutated_output = trans_layer(x)
    print("\nMutated Softmax Output:")
    print(mutated_output)

    
    print("Are the outputs equal?", np.allclose(original_output.asnumpy(), mutated_output.asnumpy(), atol=1e-5))