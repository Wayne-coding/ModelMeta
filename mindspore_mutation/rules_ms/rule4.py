
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
import numpy as np

class TransLayerRule4(nn.Cell):
    def __init__(self, layer_1):
        super(TransLayerRule4, self).__init__()
        
        self.num_features = layer_1.num_features
        self.eps = layer_1.eps
        self.momentum = layer_1.momentum
        self.track_running_stats = hasattr(layer_1, 'moving_mean')

        if hasattr(layer_1, 'gamma') and hasattr(layer_1, 'beta'):
            self.affine = True
            self.weight = Parameter(Tensor(layer_1.gamma.asnumpy(), mindspore.float32))
            self.bias = Parameter(Tensor(layer_1.beta.asnumpy(), mindspore.float32))
        else:
            self.affine = False

        if self.track_running_stats:
            self.running_mean = Parameter(Tensor(layer_1.moving_mean.asnumpy(), mindspore.float32), requires_grad=False)
            self.running_var = Parameter(Tensor(layer_1.moving_variance.asnumpy(), mindspore.float32), requires_grad=False)

    def construct(self, input):
        reduce_dim = [0]  
        i = 2
        while i < len(input.shape):  
            reduce_dim.append(i)
            i += 1

        if self.training or not self.track_running_stats:
            mean = ops.ReduceMean(True)(input, reduce_dim)
            variance = ops.ReduceVar(True)(input, reduce_dim, unbiased=False)
            if self.track_running_stats:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * variance
        else:
            mean = self.running_mean
            variance = self.running_var

        shape = (1, -1) + (1,) * (input.dim() - 2)  
        mean = ops.Reshape()(mean, shape)
        variance = ops.Reshape()(variance, shape)
        if self.affine:
            weight = ops.Reshape()(self.weight, shape)
            bias = ops.Reshape()(self.bias, shape)
        else:
            weight = Tensor(1.0, mindspore.float32)
            bias = Tensor(0.0, mindspore.float32)

        output = (input - mean) / ops.Sqrt()(variance + self.eps) * weight + bias
        return output

if __name__ == "__main__" and False:
    
    bn2 = nn.BatchNorm2d(num_features=3)
    bn2.set_train(False)  

    
    custom_bn2 = TransLayerRule4(bn2)
    custom_bn2.set_train(False)  

    
    input_tensor = Tensor(np.random.randn(8, 3, 32, 32), mindspore.float32)

    
    output_mindspore = bn2(input_tensor)

    
    output_custom = custom_bn2(input_tensor)

    
    are_outputs_close = np.allclose(output_mindspore.asnumpy(), output_custom.asnumpy(), atol=1e-6)
    print("Are the outputs close? ", are_outputs_close)

    print(bn2)
    
    print(custom_bn2)