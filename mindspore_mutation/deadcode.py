"""
deadcode1:SELayer —— ReLU() Hardsigmoid()
deadcode2:DenseLayer —— ReLU() 
deadcode3:Inception_A —— ReLU() AvgPool2d()
deadcode4:PWDWPW_ResidualBlock —— ReLU6() 
deadcode5:ResidualBlock —— ReLU() 
deadcode6:DropPath —— 无
deadcode7:Dense —— 无
"""
import collections
import mindspore
import numpy as np
from mindspore import nn, ops as ops
from mindspore.ops import operations as P
import mindspore.context as context
import mindspore as ms
from mindspore.rewrite import SymbolTree
from mindspore.rewrite import ScopedValue, NodeType
from mindspore.rewrite.node import Node, NodeManager

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")


class GlobalAvgPooling(nn.Cell):
    """
    Global avg pooling definition.
    """

    def __init__(self, keep_dims=False):
        super(GlobalAvgPooling, self).__init__()
        self.keep_dims = keep_dims
        self.mean = ops.mean

    def construct(self, x):
        dtype = x.dtype
        x = ops.cast(x, mindspore.float32)
        x = self.mean(x, (2, 3), self.keep_dims)
        x = ops.cast(x, dtype)
        return x

class SELayer(nn.Cell):
    """
    SE warpper definition.

    Args:
        num_out (int): Numbers of output channels.
        ratio (int): middle output ratio.

    Returns:
        Tensor, output tensor.

    """

    def __init__(self, ratio=1):
        super(SELayer, self).__init__()
        self.ratio = ratio
        self.SE_pool = GlobalAvgPooling(keep_dims=True)
        self.SE_act1 = self.Activation('relu')
        self.SE_act2 = self.Activation('hsigmoid')
        self.SE_mul = ops.Mul()

    @staticmethod
    def _make_divisible(x, divisor=8):
        return int(np.ceil(x * 1. / divisor) * divisor)

    def Activation(self, act_func):
        if act_func == 'relu':
            self.act = nn.ReLU()
        elif act_func == 'relu6':
            self.act = nn.ReLU6()
        elif act_func in ('hsigmoid', 'hard_sigmoid'):
            self.act = nn.HSigmoid()
        elif act_func in ('hswish', 'hard_swish'):
            self.act = nn.HSwish()
        else:
            raise NotImplementedError
        return self.act

    def construct(self, x):
        out = self.SE_pool(x)
        
        dtype = out.dtype
        conv2out = mindspore.Tensor(np.random.randn(out.shape[0], out.shape[1], 1, 1).astype(np.float32),
                                    mindspore.float32)
        out = ops.cast(out, mindspore.float32)
        out = ops.conv2d(out, weight=conv2out,
                         pad_mode='pad')
        out = ops.cast(out, dtype)
        
        out = self.SE_act1(out)
        dtype = out.dtype
        out = ops.cast(out, mindspore.float32)
        conv2out_1 = mindspore.Tensor(np.random.randn(out.shape[0], out.shape[1], 1, 1).astype(np.float32),
                                      mindspore.float32)
        out = ops.conv2d(out, weight=conv2out_1,
                         pad_mode='pad')
        out = ops.cast(out, dtype)
        out = self.SE_act2(out)
        
        return out



class DenseLayer(nn.Cell):
    def __init__(self):
        super(DenseLayer, self).__init__()
        self.drop_rate = 0.5
        self.relu = nn.ReLU()
        self.relu_1 = nn.ReLU()

    def construct(self, x):
        dtype = x.dtype
        x = ops.cast(x, mindspore.float32)
        in_shape = x.shape[1]
        new_features = nn.BatchNorm2d(in_shape)(x)
        new_features = self.relu(new_features)
        feature_weight = mindspore.Tensor(
            np.random.randn(new_features.shape[0], new_features.shape[1], 1, 1).astype(np.float32))
        new_features = ops.conv2d(new_features, feature_weight, stride=1, pad_mode="same")
        in_shape_1 = new_features.shape[1]
        new_features = nn.BatchNorm2d(in_shape_1)(new_features)
        new_features = self.relu_1(new_features)
        feature_weight_1 = mindspore.Tensor(
            np.random.randn(new_features.shape[0], new_features.shape[1], 1, 1).astype(np.float32))
        new_features = ops.conv2d(new_features, feature_weight_1, stride=1, pad_mode="same")
        
        if self.drop_rate > 0:
            new_features = nn.Dropout(p=self.drop_rate)(new_features)
        new_features = ops.cast(new_features, dtype)
        x = ops.cast(x, dtype)
        return ops.Concat(1)([x, new_features])






class BasicConv2d(nn.Cell):
    def __init__(self):
        super(BasicConv2d, self).__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        feature_weight = mindspore.Tensor(
            np.random.randn(x.shape[0], x.shape[1], 1, 1).astype(np.float32))
        dtype = x.dtype
        x = ops.cast(x, mindspore.float32)
        x = ops.conv2d(x, feature_weight, stride=1, pad_mode="same")
        in_shape_1 = x.shape[1]
        x = nn.BatchNorm2d(in_shape_1)(x)
        x = self.relu(x)
        x = ops.cast(x, dtype)
        return x


class Inception_A(nn.Cell):
    def __init__(self):
        super(Inception_A, self).__init__()
        self.concat = P.Concat(axis=1)
        self.branch0 = BasicConv2d()
        self.branch1 = nn.SequentialCell([
            BasicConv2d(),
            BasicConv2d()])
        self.branch2 = nn.SequentialCell([
            BasicConv2d(),
            BasicConv2d(),
            BasicConv2d()])
        self.branch_pool = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=1),
            BasicConv2d()])

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        dtype = x.dtype
        x = ops.cast(x, mindspore.float32)
        
        branch_pool = self.branch_pool(x)
        branch_pool = ops.cast(branch_pool, dtype)
        
        out = self.concat((x0, x1, x2, branch_pool))
        return out







class dwpw_basic(nn.Cell):
    def __init__(self, in_channel, out_channel, kernel_size, stride, depthwise, activation='relu6'):
        super(dwpw_basic, self).__init__()
        self.dwpw_conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad_mode="same",
                                   group=1 if not depthwise else in_channel)
        self.dwpw_bn = nn.BatchNorm2d(out_channel)
        if activation:
            
            self.dwpw_activation = nn.ReLU6() 

    def construct(self, x):
        
        dtype = x.dtype
        x = ops.cast(x, mindspore.float32)
        x = self.dwpw_conv(x)
        

        
        
        x = ops.batch_norm(x, running_mean=mindspore.numpy.randn(x.shape[1]),
                           running_var=mindspore.numpy.randn(x.shape[1]), momentum=0.9, eps=1e-5,
                           weight=mindspore.numpy.randn(x.shape[1]),
                           bias=mindspore.numpy.randn(x.shape[1]))
        

        x = self.dwpw_activation(x)
        x = ops.cast(x, dtype)
        
        return x

class PWDWPW_ResidualBlock(nn.Cell):
    """
    Pointwise - -Depthwise - -Pointwise - -Add
    """

    def __init__(self):
        super(PWDWPW_ResidualBlock, self).__init__()

        self.PDP_ResidualBlock_3 = None
        self.PDP_ResidualBlock_2 = None
        self.PDP_ResidualBlock_1 = None
        self.add = P.Add()

    def construct(self, x):
        identity = x
        in_channel = x.shape[1]
        self.PDP_ResidualBlock_1 = dwpw_basic(in_channel, in_channel, 1, 1, False, 'relu6')
        out1 = self.PDP_ResidualBlock_1(x)
        in_channel = out1.shape[1]

        self.PDP_ResidualBlock_2 = dwpw_basic(in_channel, in_channel, 1, 1, True, 'relu6')

        out2 = self.PDP_ResidualBlock_2(out1)
        in_channel = out2.shape[1]

        self.PDP_ResidualBlock_3 = dwpw_basic(in_channel, in_channel, 1, 1, False, 'relu6')

        out2 = self.PDP_ResidualBlock_3(out2)
        out = self.add(out2, identity)
        return out







def _conv3x3(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    """_conv3x3"""
    if res_base:
        return nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
            pad_mode="pad",
        )
    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=3,
        stride=stride,
        padding=0,
        pad_mode="same",
    )


def _conv1x1(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    """_conv1x1"""
    if res_base:
        return nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=1,
            stride=stride,
            padding=0,
            pad_mode="pad",
        )
    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=1,
        stride=stride,
        padding=0,
        pad_mode="same",
    )


def _conv7x7(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    """_conv7x7"""
    if res_base:
        return nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=7,
            stride=stride,
            padding=3,
            pad_mode="pad",
        )
    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=7,
        stride=stride,
        padding=0,
        pad_mode="same",
    )


def _bn(channel, res_base=False):
    """_bn"""
    if res_base:
        return nn.BatchNorm2d(
            channel,
            eps=1e-5,
            momentum=0.1,
            gamma_init=1,
            beta_init=0,
            moving_mean_init=0,
            moving_var_init=1,
        )
    return nn.BatchNorm2d(
        channel,
        eps=1e-4,
        momentum=0.9,
        gamma_init=1,
        beta_init=0,
        moving_mean_init=0,
        moving_var_init=1,
    )


def _bn_last(channel):
    """_bn_last"""
    return nn.BatchNorm2d(
        channel,
        eps=1e-4,
        momentum=0.9,
        gamma_init=0,
        beta_init=0,
        moving_mean_init=0,
        moving_var_init=1,
    )


def _fc(in_channel, out_channel, use_se=False):
    """_fc"""
    return nn.Dense(
        in_channel, out_channel, has_bias=True, bias_init=0,  
    )

class ResidualBlock(nn.Cell):
    expansion = 4

    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.use_se = False
        self.se_block = False

        self.residual_relu1 = nn.ReLU()
        self.residual_relu2 = nn.ReLU()
        self.residual_relu3 = nn.ReLU()

        self.residual_down_sample_layer = None

    def construct(self, x):
        dtype = x.dtype
        x = ops.cast(x, mindspore.float32)
        identity = x
        in_channel = x.shape[1]
        self.residual_conv1 = _conv1x1(in_channel, in_channel, stride=1, use_se=self.use_se)
        out = self.residual_conv1(x)
        in_channel = out.shape[1]
        self.residual_bn1 = _bn(in_channel)

        out = self.residual_bn1(out)
        out = self.residual_relu1(out)
        in_channel = out.shape[1]
        self.residual_conv2 = _conv3x3(in_channel, in_channel, stride=1, use_se=self.use_se)
        out = self.residual_conv2(out)
        in_channel = out.shape[1]

        self.residual_bn2 = _bn(in_channel)

        out = self.residual_bn2(out)
        out = self.residual_relu2(out)
        in_channel = out.shape[1]

        self.residual_conv3 = _conv1x1(in_channel, in_channel, stride=1, use_se=self.use_se)

        out = self.residual_conv3(out)
        out_channel = out.shape[1]
        self.residual_bn3 = _bn(out_channel)
        out = self.residual_bn3(out)
        in_channel = out.shape[1]
        self.residual_down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, in_channel, 1,
                                                                      use_se=self.use_se), _bn(in_channel)])
        identity = self.residual_down_sample_layer(identity)
        out = out + identity
        out = self.residual_relu3(out)
        out = ops.cast(out, dtype)
        return out






class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=0.5, seed=0):
        super(DropPath, self).__init__()
        self.keep_prob = 1 - drop_prob
        seed = min(seed, 0)  
        self.rand = P.UniformReal(seed=seed)  
        self.shape = P.Shape()
        self.floor = P.Floor()

    def construct(self, x):
        if self.training:
            x_shape = self.shape(x)  
            random_tensor = self.rand((x_shape[0], 1, 1)) if len(x_shape) == 3 else self.rand((x_shape[0], 1, 1, 1))
            random_tensor = random_tensor + self.keep_prob
            random_tensor = self.floor(random_tensor)
            x = x / self.keep_prob
            x = x * random_tensor
        return x






class op_mul(nn.Cell):
    def __init__(self):
        super(op_mul, self).__init__()
        ops_mul = ops.Mul()

    def construct(self, deada, for_matmul_edge):
        return ops_mul(deada, for_matmul_edge)

class Dense(nn.Cell):
    def __init__(self):
        super(Dense, self).__init__()

    def construct(self, deada):
        feature_a = deada.shape[-2]
        feature_b = deada.shape[-1]
        for_matmul_edge = mindspore.numpy.randn(feature_a, feature_b)
        matmul_edge = ops.Mul()(deada, for_matmul_edge)
        for_add_edge = mindspore.numpy.randn(feature_a, feature_b)
        add_edge = ops.Add()(matmul_edge, for_add_edge)
        return add_edge






banned_ops = [mindspore.ops.operations.array_ops.Shape,
              mindspore.ops.operations.array_ops.Concat,
              type(None)
              ]
banned_cell = [mindspore.nn.layer.CentralCrop, ]
banned_trees = [mindspore.ops.ResizeBilinear, 
                mindspore.ops.operations.Shape,
                type(None)
                ]
























def has_child_node(net, node_name):
    layers = net.cells_and_names()
    parent_node = None
    for name, _ in layers:
        if name == node_name:
            parent_node = name
            continue
        if parent_node is not None and name.startswith(parent_node + '.'):
            return True
    return False

if __name__ == '__main__':
    nodedict = collections.OrderedDict()  
    hash_table = collections.defaultdict(int)  
    stree= SymbolTree.create(Dense())
    
    
    
        
        
    

    
    
    
    
    
    
    

    
    
    op_layer = op_mul()

    
    

    
    

    
    
    
    

    
    for name, layer in op_layer._cells.items():
        print(f"Layer Name: {name}, Layer: {layer}")
    
    
    
    

"""
deadcode1:SELayer —— ReLU() Hardsigmoid() /
{<class 'mindspore.ops.operations.manually_defined.ops_def.Cast'>, <class 'mindspore.ops.operations.nn_ops.Conv2D'>, <class 'mindspore.nn.layer.activation.HSigmoid'>, <class 'mindspore.nn.layer.activation.ReLU'>, <class 'mindspore.ops.auto_generate.gen_ops_prim.ReduceMean'>, <class 'mindspore.ops.auto_generate.gen_ops_prim.BiasAdd'>
deadcode2:DenseLayer —— ReLU() /
{<class 'mindspore.ops.operations.manually_defined.ops_def.Cast'>, <class 'mindspore.ops.auto_generate.gen_ops_prim.BiasAdd'>, <class 'mindspore.nn.layer.activation.ReLU'>, <class 'mindspore.ops.operations.nn_ops.Conv2D'>}
deadcode3:Inception_A —— ReLU() AvgPool2d() /
{<class 'mindspore.ops.auto_generate.gen_ops_prim.BiasAdd'>, <class 'mindspore.nn.layer.activation.ReLU'>, <class 'mindspore.nn.layer.pooling.AvgPool2d'>, <class 'mindspore.ops.operations.nn_ops.Conv2D'>, <class 'mindspore.ops.operations.manually_defined.ops_def.Cast'>}
deadcode4:PWDWPW_ResidualBlock —— ReLU6() /
{<class 'mindspore.ops.auto_generate.gen_ops_prim.Add'>}，没有解决
deadcode5:ResidualBlock —— ReLU() /
{<class 'mindspore.ops.operations.manually_defined.ops_def.Cast'>, <class 'mindspore.nn.layer.activation.ReLU'>}
deadcode6:DropPath —— 无 / <class 'mindspore.ops.auto_generate.gen_ops_prim.Floor'>
deadcode7:Dense —— 无 / 无
"""