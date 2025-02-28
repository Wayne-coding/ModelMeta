import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import jit
from mindspore.ops import functional as F
import numpy as np

class TransLayerRule1Conv2d(nn.Cell):
    def __init__(self, layer):
        super(TransLayerRule1Conv2d, self).__init__()
        if not isinstance(layer, nn.Cell):
            raise ValueError("This wrapper only supports nn.Cell layers")
        self.optimized_layer = layer

    def construct(self, x):
        return self.optimized_layer(x)



class TransLayerRule1AvgPool2d(nn.Cell):
    def __init__(self, layer):
        super(TransLayerRule1AvgPool2d, self).__init__()
        if not isinstance(layer, nn.Cell):
            raise ValueError("This wrapper only supports nn.Cell layers")
        self.optimized_layer = layer

    def construct(self, x):
        return self.optimized_layer(x)

class TransLayerRule1MaxPool2d(nn.Cell):
    def __init__(self, layer):
        super(TransLayerRule1MaxPool2d, self).__init__()
        if not isinstance(layer, nn.Cell):
            raise ValueError("This wrapper only supports nn.Cell layers")
        self.optimized_layer = layer

    def construct(self, x):
        return self.optimized_layer(x)

class TransLayerRule1ReLU(nn.Cell):
    def __init__(self, layer):
        super(TransLayerRule1ReLU, self).__init__()
        if not isinstance(layer, nn.Cell):
            raise ValueError("This wrapper only supports nn.Cell layers")
        self.optimized_layer = layer

    def construct(self, x):
        return self.optimized_layer(x)

class TransLayerRule1ReLU6(nn.Cell):
    def __init__(self, layer):
        super(TransLayerRule1ReLU6, self).__init__()
        if not isinstance(layer, nn.Cell):
            raise ValueError("This wrapper only supports nn.Cell layers")
        self.optimized_layer = layer

    def construct(self, x):
        return self.optimized_layer(x)

class TransLayerRule1BatchNorm2d(nn.Cell):
    def __init__(self, layer):
        super(TransLayerRule1BatchNorm2d, self).__init__()
        if not isinstance(layer, nn.Cell):
            raise ValueError("This wrapper only supports nn.Cell layers")
        self.optimized_layer = layer

    def construct(self, x):
        return self.optimized_layer(x)

class TransLayerRule1Linear(nn.Cell):
    def __init__(self, layer):
        super(TransLayerRule1Linear, self).__init__()
        if not isinstance(layer, nn.Cell):
            raise ValueError("This wrapper only supports nn.Cell layers")
        self.optimized_layer = layer

    def construct(self, x):
        return self.optimized_layer(x)

class TransLayerRule1Flatten(nn.Cell):
    def __init__(self, layer):
        super(TransLayerRule1Flatten, self).__init__()
        if not isinstance(layer, nn.Cell):
            raise ValueError("This wrapper only supports nn.Cell layers")
        self.optimized_layer = layer

    def construct(self, x):
        return self.optimized_layer(x)

class TransLayerRule1Hardsigmoid(nn.Cell):
    def __init__(self, layer):
        super(TransLayerRule1Hardsigmoid, self).__init__()
        if not isinstance(layer, nn.Cell):
            raise ValueError("This wrapper only supports nn.Cell layers")
        self.optimized_layer = layer

    def construct(self, x):
        return self.optimized_layer(x)

class TransLayerRule1Sigmoid(nn.Cell):
    def __init__(self, layer):
        super(TransLayerRule1Sigmoid, self).__init__()
        if not isinstance(layer, nn.Cell):
            raise ValueError("This wrapper only supports nn.Cell layers")
        self.optimized_layer = layer

    def construct(self, x):
        return self.optimized_layer(x)

class TransLayerRule1Softmax(nn.Cell):
    def __init__(self, layer):
        super(TransLayerRule1Softmax, self).__init__()
        if not isinstance(layer, nn.Cell):
            raise ValueError("This wrapper only supports nn.Cell layers")
        self.optimized_layer = layer

    def construct(self, x):
        return self.optimized_layer(x)

class TransLayerRule1Tanh(nn.Cell):
    def __init__(self, layer):
        super(TransLayerRule1Tanh, self).__init__()
        if not isinstance(layer, nn.Cell):
            raise ValueError("This wrapper only supports nn.Cell layers")
        self.optimized_layer = layer

    def construct(self, x):
        return self.optimized_layer(x)

class TransLayerRule1ConvTranspose2d(nn.Cell):
    def __init__(self, layer):
        super(TransLayerRule1ConvTranspose2d, self).__init__()
        if not isinstance(layer, nn.Cell):
            raise ValueError("This wrapper only supports nn.Cell layers")
        self.optimized_layer = layer

    def construct(self, x):
        return self.optimized_layer(x)

class TransLayerRule1LeakyReLU(nn.Cell):
    def __init__(self, layer):
        super(TransLayerRule1LeakyReLU, self).__init__()
        if not isinstance(layer, nn.Cell):
            raise ValueError("This wrapper only supports nn.Cell layers")
        self.optimized_layer = layer

    def construct(self, x):
        return self.optimized_layer(x)

class TransLayerRule1AdaptiveAvgPool2d(nn.Cell):
    def __init__(self, layer):
        super(TransLayerRule1AdaptiveAvgPool2d, self).__init__()
        if not isinstance(layer, nn.Cell):
            raise ValueError("This wrapper only supports nn.Cell layers")
        self.optimized_layer = layer

    def construct(self, x):
        return self.optimized_layer(x)

class TransLayerRule1Dropout(nn.Cell):
    def __init__(self, layer):
        super(TransLayerRule1Dropout, self).__init__()
        if not isinstance(layer, nn.Cell):
            raise ValueError("This wrapper only supports nn.Cell layers")
        self.optimized_layer = layer

    def construct(self, x):
        return self.optimized_layer(x)

class TransLayerRule1Embedding(nn.Cell):
    def __init__(self, layer):
        super(TransLayerRule1Embedding, self).__init__()
        if not isinstance(layer, nn.Cell):
            raise ValueError("This wrapper only supports nn.Cell layers")
        self.optimized_layer = layer

    def construct(self, x):
        return self.optimized_layer(x)

class TransLayerRule1LSTM(nn.Cell):
    def __init__(self, layer):
        super(TransLayerRule1LSTM, self).__init__()
        if not isinstance(layer, nn.Cell):
            raise ValueError("This wrapper only supports nn.Cell layers")
        self.optimized_layer = layer

    def construct(self, x):
        return self.optimized_layer(x)

if __name__ == "__main__" and False:
    # 测试所有的转换类
    layers = [
        (nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), pad_mode='pad', padding=1), TransLayerRule1Conv2d),
        (nn.AvgPool2d(kernel_size=2, stride=2), TransLayerRule1AvgPool2d),
        (nn.MaxPool2d(kernel_size=2, stride=2), TransLayerRule1MaxPool2d),
        (nn.ReLU(), TransLayerRule1ReLU),
        (nn.ReLU6(), TransLayerRule1ReLU6),
        (nn.BatchNorm2d(64), TransLayerRule1BatchNorm2d),
        (nn.Dense(64 * 32 * 32, 128), TransLayerRule1Linear),
        (nn.Flatten(), TransLayerRule1Flatten),
        (nn.HSigmoid(), TransLayerRule1Hardsigmoid),
        (nn.Sigmoid(), TransLayerRule1Sigmoid),
        (nn.Softmax(), TransLayerRule1Softmax),
        (nn.Tanh(), TransLayerRule1Tanh),
        (nn.Conv2dTranspose(64, 128, kernel_size=(3, 3)), TransLayerRule1ConvTranspose2d),
        (nn.LeakyReLU(), TransLayerRule1LeakyReLU),
        (nn.AdaptiveAvgPool2d((1, 1)), TransLayerRule1AdaptiveAvgPool2d),
        (nn.Dropout(0.5), TransLayerRule1Dropout),
        (nn.Embedding(10, 3), TransLayerRule1Embedding),
        (nn.LSTM(10, 20), TransLayerRule1LSTM)
    ]

    # 测试输入
    input_tensor = mindspore.Tensor(ops.ones((5, 64, 32, 32)), mindspore.float32)
    for layer, wrapper_class in layers:
        optimized_layer = wrapper_class(layer)
        try:
            if isinstance(layer, nn.Embedding):
                input_tensor = mindspore.Tensor(np.random.randint(0, 10, (5, 32)), mindspore.int32)
            elif isinstance(layer, nn.LSTM):
                input_tensor = mindspore.Tensor(ops.ones((5, 32, 10)), mindspore.float32)
                output_before, _ = layer(input_tensor)
                output_after, _ = optimized_layer(input_tensor)
            elif isinstance(layer, nn.Dense):
                input_tensor = mindspore.Tensor(ops.ones((5, 64 * 32 * 32)), mindspore.float32)
            else:
                input_tensor = mindspore.Tensor(ops.ones((5, 64, 32, 32)), mindspore.float32)

            if not isinstance(layer, nn.LSTM):
                output_before = layer(input_tensor)
                output_after = optimized_layer(input_tensor)

            are_outputs_close = np.allclose(output_before.asnumpy(), output_after.asnumpy(), atol=1e-6)
            print(f"{wrapper_class.__name__} - Are outputs close? {are_outputs_close}")
        except Exception as e:
            print(f"{wrapper_class.__name__} - Error during testing: {e}")