import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor, dtype as mstype
import mindspore.numpy as mnp
import numpy as np

class Conv2dToConv3d(nn.Cell):
    def __init__(self, layer_2d):
        super(Conv2dToConv3d, self).__init__()
        self.layer_2d = layer_2d
        new_kernel_size = (1, self.layer_2d.kernel_size, self.layer_2d.kernel_size) if isinstance(self.layer_2d.kernel_size, int) else (1, *self.layer_2d.kernel_size)
        new_stride = (1, self.layer_2d.stride, self.layer_2d.stride) if isinstance(self.layer_2d.stride, int) else (1, *self.layer_2d.stride)
        new_padding = (0, 0, self.layer_2d.padding, self.layer_2d.padding, self.layer_2d.padding, self.layer_2d.padding) if isinstance(self.layer_2d.padding, int) else (0, 0, *self.layer_2d.padding, *self.layer_2d.padding)
        new_dilation = (1, self.layer_2d.dilation, self.layer_2d.dilation) if isinstance(self.layer_2d.dilation, int) else (1, *self.layer_2d.dilation)

        self.layer_3d = nn.Conv3d(
            in_channels=self.layer_2d.in_channels,
            out_channels=self.layer_2d.out_channels,
            kernel_size=new_kernel_size,
            stride=new_stride,
            pad_mode='pad',
            padding=new_padding,
            dilation=new_dilation,
            has_bias=self.layer_2d.has_bias
        )
        self.layer_3d.weight = Parameter(mnp.expand_dims(self.layer_2d.weight.data, 2))
        if self.layer_2d.has_bias:
            self.layer_3d.bias = Parameter(self.layer_2d.bias.data)

    def construct(self, x):
        x = ops.ExpandDims()(x, 2)
        x = self.layer_3d(x)
        x = ops.Squeeze(2)(x)
        return x

class MaxPool2dToMaxPool3d(nn.Cell):
    def __init__(self, layer_2d):
        super(MaxPool2dToMaxPool3d, self).__init__()
        self.layer_2d = layer_2d
        new_kernel_size = (1, self.layer_2d.kernel_size, self.layer_2d.kernel_size) if isinstance(self.layer_2d.kernel_size, int) else (1, *self.layer_2d.kernel_size)
        new_stride = (1, self.layer_2d.stride, self.layer_2d.stride) if isinstance(self.layer_2d.stride, int) else (1, *self.layer_2d.stride)

        self.layer_3d = nn.MaxPool3d(
            kernel_size=new_kernel_size,
            stride=new_stride,
            pad_mode='pad'
        )

    def construct(self, x):
        x = ops.ExpandDims()(x, 2)
        x = self.layer_3d(x)
        x = ops.Squeeze(2)(x)
        return x

class AvgPool2dToAvgPool3d(nn.Cell):
    def __init__(self, layer_2d):
        super(AvgPool2dToAvgPool3d, self).__init__()
        self.layer_2d = layer_2d
        new_kernel_size = (1, self.layer_2d.kernel_size, self.layer_2d.kernel_size) if isinstance(self.layer_2d.kernel_size, int) else (1, *self.layer_2d.kernel_size)
        new_stride = (1, self.layer_2d.stride, self.layer_2d.stride) if isinstance(self.layer_2d.stride, int) else (1, *self.layer_2d.stride)

        self.layer_3d = nn.AvgPool3d(
            kernel_size=new_kernel_size,
            stride=new_stride,
            pad_mode='pad'
        )

    def construct(self, x):
        x = ops.ExpandDims()(x, 2)
        x = self.layer_3d(x)
        x = ops.Squeeze(2)(x)
        return x

if __name__ == "__main__":
    # 示例输入
    input_2d = Tensor(np.random.randn(1, 3, 224, 224), mstype.float32)  # 输入为一个2D图像

    # Conv2d 测试
    conv2d = nn.Conv2d(3, 64, kernel_size=3, stride=1, pad_mode='pad', padding=1)
    conv_wrapper = Conv2dToConv3d(conv2d)
    print(conv_wrapper)
    output = conv_wrapper(input_2d)
    print("Output shape after Conv2dTo3dWrapper (Conv2d):", output.shape)
    output1 = conv2d(input_2d)
    print("Original Conv2d output shape:", output1.shape)
    print(mnp.isclose(output, output1, atol=1e-6).all())  # 允许一定的误差范围来比较结果

    # MaxPool2d 测试
    maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='pad')
    maxpool_wrapper = MaxPool2dToMaxPool3d(maxpool2d)
    print(maxpool_wrapper)
    output = maxpool_wrapper(input_2d)
    print("Output shape after MaxPool2dToMaxPool3d (MaxPool2d):", output.shape)
    output1 = maxpool2d(input_2d)
    print("Original MaxPool2d output shape:", output1.shape)
    print(mnp.isclose(output, output1, atol=1e-6).all())  # 允许一定的误差范围来比较结果

    # AvgPool2d 测试
    avgpool2d = nn.AvgPool2d(kernel_size=2, stride=2, pad_mode='pad')
    avgpool_wrapper = AvgPool2dToAvgPool3d(avgpool2d)
    print(avgpool_wrapper)
    output = avgpool_wrapper(input_2d)
    print("Output shape after AvgPool2dToAvgPool3d (AvgPool2d):", output.shape)
    output1 = avgpool2d(input_2d)
    print("Original AvgPool2d output shape:", output1.shape)
    print(mnp.isclose(output, output1, atol=1e-6).all())  # 允许一定的误差范围来比较结果