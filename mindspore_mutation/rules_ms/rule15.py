import mindspore
import mindspore.nn as nn
import numpy as np
from mindspore import Tensor, context

DELTA = 10

class TransLayerRule15ReLU(nn.Cell):
    def __init__(self, layer_relu):
        super(TransLayerRule15ReLU, self).__init__()

        self.layer_relu = layer_relu
        self.delta = Tensor(np.random.uniform(0, DELTA, 1)[0], mindspore.float32)

    def construct(self, x):
        mut_x = x * self.delta
        return self.layer_relu(mut_x) / self.delta

class TransLayerRule15LeakyReLU(nn.Cell):
    def __init__(self, layer_relu):
        super(TransLayerRule15LeakyReLU, self).__init__()

        self.layer_relu = layer_relu
        self.delta = Tensor(np.random.uniform(0, DELTA, 1)[0], mindspore.float32)

    def construct(self, x):
        mut_x = x * self.delta
        return self.layer_relu(mut_x) / self.delta

if __name__ == "__main__" and False:
    
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    relu = nn.LeakyReLU()
    relu_standard = nn.ReLU()

    
    trans_layer_leaky_relu = TransLayerRule15LeakyReLU(relu)
    trans_layer_relu = TransLayerRule15ReLU(relu_standard)

    
    x = Tensor(np.random.randn(5, 10), mindspore.float32)  

    
    original_output_relu = relu_standard(x)
    original_output_leaky_relu = relu(x)

    
    mutated_output_relu = trans_layer_relu(x)
    mutated_output_leaky_relu = trans_layer_leaky_relu(x)

    
    diff_relu = mindspore.ops.Abs()(original_output_relu - mutated_output_relu).max()
    diff_leaky_relu = mindspore.ops.Abs()(original_output_leaky_relu - mutated_output_leaky_relu).max()

    
    print("\nMaximum output difference for ReLU:", diff_relu.asnumpy())
    print("\nMaximum output difference for LeakyReLU:", diff_leaky_relu.asnumpy())