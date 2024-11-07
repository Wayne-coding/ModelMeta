













import os
import shutil
from mindspore import context
from mindspore import load_checkpoint

from model_utils.device_adapter import get_device_id
from model_utils.config import config as cfg
from src.network import AutoEncoder
from src.eval_utils import apply_eval
from src.utils import get_results

context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target, device_id=get_device_id())


def get_network():
    current_path = os.path.abspath(os.path.dirname(__file__))
    auto_encoder = AutoEncoder(cfg)
    if cfg.model_arts:
        import moxing as mox

        mox.file.copy_parallel(src_url=cfg.checkpoint_url, dst_url=cfg.cache_ckpt_file)
        ckpt_path = cfg.cache_ckpt_file
    else:
        ckpt_path = cfg.checkpoint_path
    ckpt_path = os.path.join(current_path, "ssimae_ascend_v190_mvtecadbottle_official_cv_ok96.8_nok96.8_avg95.2.ckpt")
    load_checkpoint(ckpt_path, net=auto_encoder)
    auto_encoder.set_train(False)
    return auto_encoder


if __name__ == "__main__":
    if os.path.exists(cfg.save_dir):
        shutil.rmtree(cfg.save_dir, True)
    net = get_network()
    get_results(cfg, net)
    print("Generate results at", cfg.save_dir)
    apply_eval(cfg)
