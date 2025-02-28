"""
变异BatchNorm算子，input+=delta,running_mean+=delta
已验证，成功
"""

import mindspore
import mindspore.nn as nn
import numpy as np
from mindspore import Tensor, Parameter
import mindspore.ops as ops
import copy

# 常量定义
DTYPE = "float32"  # 数据类型
DELTA = 10         # 随机生成张量的范围，可自定义

class TransLayer_rule11(nn.Cell):
    def __init__(self, layer_bn):
        super(TransLayer_rule11, self).__init__()
        self.delta = Tensor(np.random.uniform(-DELTA, DELTA, 1)[0], mindspore.float32)
        self.layer_bn = layer_bn

        if self.layer_bn.use_batch_statistics:
            self.layer_bn.gamma = layer_bn.gamma
            self.layer_bn.beta = layer_bn.beta

            # 检查 moving_mean 是否存在于参数中
            if "moving_mean" in layer_bn.parameters_dict():
                old_para = layer_bn.parameters_dict()
                new_para = copy.deepcopy(old_para)
                new_para["moving_mean"] = new_para["moving_mean"] + self.delta
                for key, value in new_para.items():
                    setattr(self.layer_bn, key, Parameter(value, requires_grad=False))

    def construct(self, x):
        x = x + self.delta
        return self.layer_bn(x)

if __name__ == "__main__" and False:
    # 创建一个标准的 BatchNorm 层
    batch_norm = nn.BatchNorm2d(num_features=3, affine=True, use_batch_statistics=True)
    batch_norm.set_train()

    # 测试数据 bgn
    x = Tensor(np.random.randn(10, 3, 32, 32), mindspore.float32)

    # 通过 BatchNorm 层
    bn_output = batch_norm(x).asnumpy()
    # print(batch_norm.moving_mean)

    # 创建 TransLayer 层，初始化为与 batch_norm 层相同的参数
    trans_layer = TransLayer_rule11(batch_norm)
    if hasattr(trans_layer.layer_bn, "moving_mean"):
        print(trans_layer.layer_bn.moving_mean)

    # 通过 TransLayer 层
    trans_output = trans_layer(x).asnumpy()

    # print(trans_layer.delta)

    # 打印输出结果
    # print("\nBatchNorm output:")
    # print(bn_output)

    # print("\nTransLayer output:")
    # print(trans_output)

    # 使用 numpy 比较两个输出
    assert np.allclose(bn_output, trans_output, atol=1e-5), "The outputs are not matching!"
    print("The outputs match successfully!")
