import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import mindspore
from mindspore import Tensor, Parameter
import mindspore.common.initializer as init
from mindspore import context


DELTA = 10  

class TransLayerRule8(nn.Cell):
    def __init__(self, layer_conv):
        super(TransLayerRule8, self).__init__()
        if not isinstance(layer_conv, nn.Cell):
            raise ValueError("This wrapper only supports nn.Cell layers")

        self.layer_conv = nn.Conv2d(
            in_channels=layer_conv.in_channels,
            out_channels=layer_conv.out_channels,
            kernel_size=layer_conv.kernel_size,
            stride=layer_conv.stride,
            padding=layer_conv.padding,
            dilation=layer_conv.dilation,
            has_bias=(layer_conv.has_bias)
        )

        self.delta = Tensor(np.random.uniform(-DELTA, DELTA, 1).astype("float32"))

        
        self.layer_conv.weight.set_data(layer_conv.weight.data)
        if layer_conv.has_bias:
            self.layer_conv.bias.set_data(layer_conv.bias.data + self.delta)

    def construct(self, x):
        return self.layer_conv(x) - self.delta




if __name__ == "__main__" and False:
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    
    
    original_layer = nn.Conv2d(
        in_channels=3,
        out_channels=6,
        kernel_size=3,
        stride=1,
        padding=0,  
        has_bias=True
    )

    
    transformed_layer = TransLayerRule8(original_layer)

    
    input_data = Tensor(np.random.randn(1, 3, 32, 32).astype("float32"))

    
    original_output = original_layer(input_data).asnumpy()
    transformed_output = transformed_layer(input_data).asnumpy()

    
    assert np.allclose(original_output, transformed_output, atol=1e-6), "The outputs are not matching!"
    print("The outputs match successfully!")