import argparse
import time
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from gpt_torch import GPTPyTorch, GPTWithLoss
from src.dataset import create_dataset
from src.utils import GPTConfig


def run_train():
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
    config = GPTConfig(batch_size=1,
                       seq_length=1024,
                       vocab_size=50257,
                       embedding_size=1024,
                       num_layers=24,
                       num_heads=16,
                       expand_ratio=4,
                       post_layernorm_residual=False,
                       dropout_rate=0.1,
                       compute_dtype=torch.float32,
                       use_past=False)
    # Re-instantiating the models with the new configuration
    gpt_model_pytorch = GPTPyTorch(config).to(device)
    loss_fn = CrossEntropyLoss().to(device)
    gpt_with_loss_model_pytorch_corrected = GPTWithLoss(gpt_model_pytorch, loss_fn, eos_token=0).to(
        device)

    # Adjusting the random input data for the model to ensure it's within the valid range
    # input_data_pytorch = torch.randint(0, config.vocab_size, (1, 256))

    # Forward pass through the model with the corrected version
    # output = gpt_with_loss_model_pytorch_corrected(input_data_pytorch)
    # print(output)
    optimizer = Adam(gpt_with_loss_model_pytorch_corrected.parameters(), lr=1e-5)
    ds = create_dataset(1, data_path=args_opt.data_path, device_num=1, rank=0)
    start_time = time.time()
    ds = ds.create_tuple_iterator(output_numpy=True)
    epoch_num = 6
    for epoch in range(epoch_num):
        for data in ds:
            optimizer.zero_grad()
            end_time = time.time()
            data0 = torch.tensor(data[0], dtype=torch.int64).to(device)
            # print("data0", data0.dtype)
            # print("maximum", torch.max(data0))
            # print("minimum", torch.min(data0))
            print("time", end_time - start_time)
            print("================================================================")
            loss_torch = gpt_with_loss_model_pytorch_corrected(data0)
            loss_torch.backward()
            optimizer.step()
            print("loss_torch", loss_torch)
            print("================================================================")
            start_time = time.time()
            # break


if __name__ == "__main__":
    from gpt_torch import device
    run_train()
