import os
import argparse
import time
import mindspore
from mindspore import context
import mindspore.nn as nn
from mindspore.parallel._transformer.loss import CrossEntropyLoss
from mindspore.parallel._transformer import TransformerOpParallelConfig
import mindspore.common.dtype as mstype
from mindspore import set_seed
from src.dataset import create_dataset
from src.gpt import GPT, GPTWithLoss
from src.utils import GPTConfig


def run_train():
    """train function for GPT"""
    parser = argparse.ArgumentParser(description="GPT training")
    parser.add_argument('--device_id', type=int, default=0, help="Device id, default is 0.")
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
    args_opt.data_path = "mindb"
    device_id = int(os.getenv("DEVICE_ID", '0'))
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

    config = GPTConfig(batch_size=1,
                       seq_length=1024,
                       vocab_size=50257,
                       embedding_size=1024,
                       num_layers=24,
                       num_heads=16,
                       expand_ratio=4,
                       post_layernorm_residual=False,
                       dropout_rate=0.1,
                       compute_dtype=mstype.float16,
                       use_past=False)
    ds = create_dataset(config.batch_size, data_path=args_opt.data_path, device_num=device_num, rank=rank)
    start_time = time.time()
    ds = ds.create_tuple_iterator()
    epoch_num = 6
    for epoch in range(epoch_num):
        for data in ds:
            # print("data0", type(data[0]))
            end_time = time.time()
            print("time", end_time - start_time)
            print("================================================================")
            loss_ms = train_step(data[0])
            print("loss_ms", loss_ms)
            print("================================================================")
            start_time = time.time()


if __name__ == "__main__":
    set_seed(12315)
    run_train()
