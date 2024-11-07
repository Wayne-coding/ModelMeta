














"""Moxing adapter for ModelArts"""

import os

_GLOBAL_SYNC_COUNT = 0


def get_device_id():
    device_id = os.getenv("DEVICE_ID", "0")
    return int(device_id)


def get_device_num():
    device_num = os.getenv("RANK_SIZE", "1")
    return int(device_num)


def get_rank_id():
    global_rank_id = os.getenv("RANK_ID", "0")
    return int(global_rank_id)


def get_job_id():
    job_id = os.getenv("JOB_ID")
    job_id = job_id if job_id != "" else "default"
    return job_id
