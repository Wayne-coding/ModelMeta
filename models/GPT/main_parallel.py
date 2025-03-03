import argparse
import time
import torch
from torch.nn import CrossEntropyLoss as CrossEntropyLossTorch
from torch.optim import Adam
from network.nlp.GPT.eval import get_acc, get_acc_torch
from network.nlp.GPT.gpt_torch import GPTPyTorch, GPTWithLoss as GPTWithLossTorch, EvalNetPyTorch
import mindspore
from mindspore import context
import mindspore.nn as nn
from mindspore.parallel._transformer.loss import CrossEntropyLoss
from mindspore.parallel._transformer import TransformerOpParallelConfig
import mindspore.common.dtype as mstype
from network.nlp.GPT.src.dataset import create_dataset
from network.nlp.GPT.src.gpt import GPT, GPTWithLoss, EvalNet
from network.nlp.GPT.src.utils import GPTConfig


device = "cuda:1"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GPT training")
    parser.add_argument('--device_id', type=int, default=5, help="Device id, default is 0.")
    parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
    parser.add_argument("--distribute", type=str, default="false", choices=["true", "false"],
                        help="Run distribute, default is false.")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "lamb"],
                        help="select which optimizer to be used, default adam")
    parser.add_argument("--epoch_size", type=int, default=10, help="Epoch size, default is 10.")
    parser.add_argument("--warmup_step", type=int, default=10000, help="Warmup step, default is 10000.")
    parser.add_argument("--data_path", type=str, default="", help="Data path of your MindRecord files.")
    parser.add_argument("--start_lr", type=float, default="5e-5", help="Start learning rate, default is 5e-5.")
    parser.add_argument("--end_lr", type=float, default="1e-10", help="End learning rate, default is 1e-10.")
    parser.add_argument("--sink_size", type=int, default=100, help="Sink size for every iteration, default is 100")
    parser.add_argument("--model_parallel_num", type=int, default=8, help="Num of model parallel, default is 8")
    args_opt = parser.parse_args()
    args_opt.data_path = "/data1/myz/net-sv/network/nlp/GPT/mindb"
    device_id = 0

    config = GPTConfig(batch_size=1,
                       seq_length=1024,
                       vocab_size=50257,
                       embedding_size=1024,
                       num_layers=8,
                       num_heads=16,
                       expand_ratio=4,
                       post_layernorm_residual=False,
                       dropout_rate=0.1,
                       compute_dtype=mstype.float16,
                       use_past=False)

    context.set_context(device_target="GPU", device_id=device_id)
    rank = 0
    device_num = 1
    model = GPT()
    args_opt.model_parallel_num = 1
    model_parallel_num = args_opt.model_parallel_num
    data_parallel_num = int(device_num / model_parallel_num)
    parallel_config = TransformerOpParallelConfig(data_parallel=data_parallel_num,
                                                  model_parallel=model_parallel_num)
    loser = CrossEntropyLoss(parallel_config.dp_mp_config)
    losser = GPTWithLoss(model, loser)
    decay_filter = lambda x: 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()
    params = model.trainable_params()
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [{'params': decay_params, 'weight_decay': 1e-2},
                    {'params': other_params, 'weight_decay': 0.0},
                    {'order_params': params}]
    optimizer = nn.AdamWeightDecay(group_params, learning_rate=1e-5)

    def forward_fn(data):
        logits = model(data)
        loss = losser(data)
        return loss, logits


    grad_fn = mindspore.ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)


    def train_step(data):
        (loss, _), grads = grad_fn(data)
        loss = mindspore.ops.depend(loss, optimizer(grads))
        return loss



    ds = create_dataset(config.batch_size, data_path=args_opt.data_path, device_num=device_num, rank=rank)
    start_time = time.time()
    gpt_model_pytorch = GPTPyTorch(config).to(device)
    loss_fn = CrossEntropyLossTorch().to(device)
    gpt_with_loss_model_pytorch_corrected = GPTWithLossTorch(gpt_model_pytorch, loss_fn, eos_token=0).to(
        device)
    optimizer_torch = Adam(gpt_with_loss_model_pytorch_corrected.parameters(), lr=1e-5)
    ds = ds.create_tuple_iterator()
    epoch_num = 1
    gpt_eval = EvalNet(model, generate=False)
    gpt_eval_torch = EvalNetPyTorch(gpt_model_pytorch, generate=False)
    # load_param_into_net(gpt_eval, ckpt_dict)
    task = "lambada"
    eval_dataset = create_dataset(config.batch_size, data_path=args_opt.data_path, drop=False)
    gpt_eval.set_train(False)
    for epoch in range(epoch_num):
        for data in ds:
            end_time = time.time()
            print("torch time", end_time - start_time)
            print("================================================================")
            loss_ms = train_step(data[0])
            print("loss_ms", loss_ms)
            print("================================================================")
            start_time = time.time()
            data0 = torch.tensor(data[0].asnumpy(), dtype=torch.int64).to(device)
            print("ms time", end_time - start_time)
            print("================================================================")

            loss_torch = gpt_with_loss_model_pytorch_corrected(data0)
            loss_torch.backward()
            print("loss_torch", loss_torch)
            optimizer_torch.step()
            optimizer_torch.zero_grad()
            print("================================================================")
            break

        acc_torch = get_acc_torch(gpt_eval_torch, eval_dataset)
        acc = get_acc(gpt_eval, eval_dataset)
        print("Accuracy is ", acc)
        print("Accuracy torch is ", acc_torch)
