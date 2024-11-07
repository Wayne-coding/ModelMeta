













"""eval resnet."""
import os
import mindspore as ms
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits

from models.resnet50.resnet50 import create_cifar10_dataset

ms.set_seed(1)

























def eval_net(net):
    """eval net"""
    target = "CPU"
    
    ms.set_context(mode=ms.GRAPH_MODE, device_target=target, save_graphs=False)

    
    dataset = create_cifar10_dataset(data_home="../../datasets/cifar10", image_size=224, batch_size=32,
                                     training=False)

    

    
    param_dict = ms.load_checkpoint("/data1/CKPTS/resnet50/resnet50_ascend_v190_cifar10_official_cv_top1acc91.00.ckpt")
    ms.load_param_into_net(net, param_dict)
    net.set_train(False)

    
    
    
    
    
    
    
    
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    
    model = ms.Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

    
    res = model.eval(dataset)
    print("result:", res)




