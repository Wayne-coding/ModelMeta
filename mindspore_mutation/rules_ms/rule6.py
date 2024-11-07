import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np

DELTA = 10  

class TransLayerRule6(nn.Cell):
    def __init__(self, layer_conv):
        super(TransLayerRule6, self).__init__()
        if not isinstance(layer_conv, nn.Cell):
            raise ValueError("This wrapper only supports nn.Cell layers")

        self.layer_conv = nn.Conv2d(
            in_channels=layer_conv.in_channels,
            out_channels=layer_conv.out_channels,
            kernel_size=layer_conv.kernel_size,
            stride=layer_conv.stride,
            padding=layer_conv.padding if isinstance(layer_conv.padding, (int, tuple)) else 0,
            has_bias=layer_conv.has_bias
        )
        self.delta = Tensor(np.random.uniform(-DELTA, DELTA, 1)[0].astype("float32"))

        
        self.layer_conv.weight.set_data(layer_conv.weight.data * self.delta)
        if layer_conv.has_bias:
            self.layer_conv.bias.set_data(layer_conv.bias.data)

    def construct(self, x):
        out = self.layer_conv(x)
        if self.layer_conv.has_bias:
            bias_reshaped = self.layer_conv.bias.reshape(-1, 1, 1)
            out = (out - bias_reshaped) / self.delta + bias_reshaped
        return out

if __name__ == "__main__" and False:
    
    layer_conv = nn.Conv2d(3, 16, 3, stride=1, padding=0, has_bias=True)
    x = Tensor(np.random.randn(1, 3, 64, 64), mindspore.float32)

    
    initial_output = layer_conv(x)
    print("Initial layer output shape:", initial_output.shape)

    
    trans_layer = TransLayerRule6(layer_conv)
    trans_output = trans_layer(x)
    print("Transformed layer output shape:", trans_output.shape)

    
    print("Are the outputs equal?", np.allclose(initial_output.asnumpy(), trans_output.asnumpy(), atol=1e-5))

    
    print("delta:")
    print(trans_layer.delta.asnumpy())
    print("Original Conv Layer:")
    print(layer_conv.weight.data.asnumpy()[1])
    print("TransLayer:")
    print(trans_layer.layer_conv.weight.data.asnumpy()[1])

    
    
    
    
    
    # print(trans_layer.delta.asnumpy())