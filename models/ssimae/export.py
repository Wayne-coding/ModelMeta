














import numpy as np
from mindspore import context, Tensor
from mindspore import load_checkpoint, export

from model_utils.device_adapter import get_device_id
from model_utils.config import config as cfg
from src.network import AutoEncoder

context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target, device_id=get_device_id())


def get_network():
    auto_encoder = AutoEncoder(cfg)
    if cfg.model_arts:
        import moxing as mox

        mox.file.copy_parallel(src_url=cfg.checkpoint_url, dst_url=cfg.cache_ckpt_file)
        ckpt_path = cfg.cache_ckpt_file
    else:
        ckpt_path = cfg.checkpoint_path

    load_checkpoint(ckpt_path, net=auto_encoder)
    auto_encoder.set_train(False)
    return auto_encoder


def model_export():
    auto_encoder = get_network()
    channel = 1 if cfg.grayscale else 3
    input_size = cfg.crop_size
    batch_size = ((cfg.mask_size - cfg.crop_size) // cfg.stride + 1) ** 2
    input_data = Tensor(np.ones([batch_size, channel, input_size, input_size], np.float32))
    export(auto_encoder, input_data, file_name=f"SSIM-AE-{cfg.dataset}", file_format="MINDIR")


if __name__ == "__main__":
    model_export()
