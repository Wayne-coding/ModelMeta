import math
import numpy as np
import torch
from torch import Tensor, nn, LongTensor,FloatTensor
from tqdm import tqdm
from src.dataset import MovieReview
from src.textcnn_torch import TextCNN
device="cpu"


def loss_com(logit, label):
    """
    construct
    """
    
    exp = torch.exp
    reduce_sum = torch.sum
    onehot = nn.functional.one_hot
    
    
    div = torch.div
    log = torch.log
    sum_cross_entropy = torch.sum
    mul = torch.Tensor.mul
    reduce_mean = torch.std_mean
    reduce_max = torch.max  
    sub = torch.Tensor.sub
    logit_max, _ = reduce_max(logit, -1, keepdim=True)
    
    exp0 = exp(sub(logit, logit_max))
    
    exp_sum = reduce_sum(exp0, -1, keepdim=True)
    
    softmax_result = div(exp0, exp_sum)
    
    label = onehot(label, num_classes)
    
    softmax_result_log = log(softmax_result)
    
    loss = sum_cross_entropy((mul(softmax_result_log, label)), -1, keepdim=False)
    
    loss = mul(Tensor([-1.0]).to(device), loss)
    
    loss, _ = reduce_mean(loss, -1, keepdim=False)
    return loss


if __name__ == '__main__':
    epoch_num, batch_size = 5, 64
    num_classes=2
    instance = MovieReview(root_dir="data/rt-polaritydata", maxlen=51, split=0.9)
    train_dataset = instance.create_train_dataset(batch_size=batch_size, epoch_size=epoch_num)
    test_dataset = instance.create_train_dataset(batch_size=batch_size, epoch_size=epoch_num)
    net = TextCNN(vocab_len=instance.get_dict_len(), word_len=51,
                  num_classes=num_classes, vec_length=40).to(device)
    train_iter = train_dataset.create_dict_iterator(output_numpy=False, num_epochs=epoch_num)
    test_iter = test_dataset.create_dict_iterator(output_numpy=False, num_epochs=epoch_num)
    opt = torch.optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=float(1e-3),
                           weight_decay=float(3e-5))

    from torchsummaryDemo import summary

    summary(net,(51,))

    for epoch in range(epoch_num):
        net.train()
        print("epoch", epoch, "/", epoch_num)
        batch = 0
        for item in train_iter:
            text_array, targets_array = item['data'].numpy(), item['label'].numpy()
            print(text_array.shape)
            print(targets_array.shape)

            break
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



