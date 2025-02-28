import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np

class TransLayerRule2(nn.Cell):
    def __init__(self, layer_2d):
        super(TransLayerRule2, self).__init__()
        if not isinstance(layer_2d, nn.Conv2d):
            raise ValueError("This wrapper only supports Conv2d layers")

        # 修改传入的 Conv2d 层，将其 padding 参数设为 0
        self.layer_2d = nn.Conv2d(
            in_channels=layer_2d.in_channels,
            out_channels=layer_2d.out_channels,
            kernel_size=layer_2d.kernel_size,
            stride=layer_2d.stride,
            pad_mode='valid',  # 强制设置 padding 为 0
            dilation=layer_2d.dilation,
            has_bias=layer_2d.has_bias
        )

        # 复制原始层的权重和偏置
        self.layer_2d.weight.set_data(layer_2d.weight)
        if layer_2d.has_bias:
            self.layer_2d.bias.set_data(layer_2d.bias)

    def construct(self, x):
        # 获取卷积核大小、步幅和膨胀系数
        kernel_size = self.layer_2d.kernel_size
        stride = self.layer_2d.stride
        dilation = self.layer_2d.dilation

        # 计算所需的填充大小
        padding_h = self._calculate_padding(x.shape[2], stride[0], kernel_size[0], dilation[0])
        padding_w = self._calculate_padding(x.shape[3], stride[1], kernel_size[1], dilation[1])

        # 使用 ops.pad 对输入进行填充
        pad = ops.Pad(((0, 0), (0, 0), (padding_h // 2, padding_h - padding_h // 2), (padding_w // 2, padding_w - padding_w // 2)))
        x = pad(x)

        # 通过 Conv2d 层
        x = self.layer_2d(x)
        return x

    def _calculate_padding(self, input_size, stride, kernel_size, dilation):
        output_size = (input_size + stride - 1) // stride  # 向上取整
        total_padding = max((output_size - 1) * stride + (kernel_size - 1) * dilation + 1 - input_size, 0)
        return total_padding

if __name__ == "__main__" and False:
    # 示例：使用 TransLayerRule2 包装一个 Conv2d
    conv2d = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), pad_mode='same')
    same_padding_conv_wrapper = TransLayerRule2(conv2d)

    input_2d = Tensor(np.random.randn(1, 3, 224, 224), mindspore.float32)  # 输入为一个2D图像
    
    # 使用原始层进行前向传播
    original_output = conv2d(input_2d)

    # 使用包装后的层进行前向传播
    output = same_padding_conv_wrapper(input_2d)
    print(original_output.shape)
    print(output.shape)

    # 验证改变前后，层的输出是否一致
    inequality_mask = np.abs(original_output.asnumpy() - output.asnumpy()) > 1e-5
    inequality_count = np.sum(inequality_mask)
    print(f"不相等的元素数量: {inequality_count}")
    