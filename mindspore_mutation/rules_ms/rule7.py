import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np


class TransLayerRule7(nn.Cell):
    def __init__(self, layer_conv):
        super(TransLayerRule7, self).__init__()
        
        
        self.layer_conv = nn.Conv2d(
            in_channels=layer_conv.in_channels,
            out_channels=layer_conv.out_channels,
            kernel_size=layer_conv.kernel_size,
            stride=layer_conv.stride,
            pad_mode='pad',
            padding=layer_conv.padding,
            dilation=layer_conv.dilation,
            group=1,  
            has_bias=(layer_conv.bias is not None)
        )

        
        self.layer_conv.weight.set_data(layer_conv.weight)
        if layer_conv.bias is not None:
            self.layer_conv.bias.set_data(layer_conv.bias)

    def construct(self, x):
        
        conv_output = self.layer_conv(x)
        if self.layer_conv.has_bias:
            bias_reshaped = self.layer_conv.bias.reshape(1, -1, 1, 1)
            conv_output = conv_output - bias_reshaped + bias_reshaped
        return conv_output



if __name__ == "__main__" and False:
    
    mindspore_layer_original = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True)

    
    mindspore_layer_trans = TransLayerRule7(mindspore_layer_original)

    
    input_data = np.random.randn(1, 3, 32, 32).astype(np.float32)
    mindspore_input = Tensor(input_data)

    
    original_output = mindspore_layer_original(mindspore_input).asnumpy()

    
    trans_output = mindspore_layer_trans(mindspore_input).asnumpy()

    
    assert np.allclose(original_output, trans_output, atol=1e-5), "The outputs are not matching!"
    print("The outputs match successfully!")