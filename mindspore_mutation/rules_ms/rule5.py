# MindSpore equivalent of PyTorch code
import mindspore
import mindspore.nn as nn
import numpy as np
from mindspore import Tensor
from mindspore.common.initializer import Normal
from mindspore import Parameter
import mindspore.ops as ops

class TransLayerRule5(nn.Cell):
    def __init__(self, layer_conv):
        super(TransLayerRule5, self).__init__()
        if not isinstance(layer_conv, nn.Cell):
            raise ValueError("This wrapper only supports nn.Cell layers")

        self.layer_conv = nn.Conv2d(
            in_channels=layer_conv.in_channels,
            out_channels=layer_conv.out_channels,
            kernel_size=layer_conv.kernel_size,
            stride=layer_conv.stride,
            padding=layer_conv.padding,
            pad_mode='pad',
            weight_init=Normal(0.02)
        )

        # Copy weights and bias from the original layer
        self.layer_conv.weight.set_data(layer_conv.weight.data)
        if layer_conv.has_bias:
            self.layer_conv.bias.set_data(layer_conv.bias.data)

    def construct(self, x):
        return self.layer_conv(x)

if __name__ == "__main__" and False:
    # Testing the MindSpore implementation independently
    import mindspore.context as context
    
    # Set context to PYNATIVE_MODE for MindSpore (dynamic graph)
    context.set_context(mode=context.PYNATIVE_MODE)

    # Define the original MindSpore layer and input
    original_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, pad_mode='pad')
    original_input = Tensor(np.random.randn(1, 3, 64, 64).astype(np.float32))

    # Get the output from the original layer
    original_output = original_layer(original_input).asnumpy()

    # Define the transformed layer using TransLayerRule5
    transformed_layer = TransLayerRule5(original_layer)

    # Get the output from the transformed layer
    transformed_output = transformed_layer(original_input).asnumpy()

    # Compare the outputs
    assert np.allclose(original_output, transformed_output, atol=1e-5), "The outputs are not identical!"

    print("Transformation successful, and outputs match!")
    print(original_layer)
    print(transformed_layer)
