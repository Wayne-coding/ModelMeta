














"""Device adapter for ModelArts"""

from model_utils.config import config as cfg

if cfg.model_arts:
    from model_utils.moxing_adapter import get_device_id, get_device_num, get_rank_id, get_job_id
else:
    from model_utils.local_adapter import get_device_id, get_device_num, get_rank_id, get_job_id

__all__ = ["get_device_id", "get_device_num", "get_rank_id", "get_job_id"]
