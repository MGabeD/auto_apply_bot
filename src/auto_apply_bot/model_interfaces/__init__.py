import torch
from auto_apply_bot.utils.logger import get_logger


logger = get_logger(__name__)


def determine_batch_size(device_index: int = 0) -> int:
    """
    Dynamically determines batch size based on free GPU VRAM.
    :param device_index: CUDA device index (default = 0)
    :return: Optimal batch size
    """
    free_mem, _ = torch.cuda.mem_get_info(device_index)
    vram_gb = free_mem / 1e9
    logger.info(f"Determining batch size for {vram_gb} free GB of VRAM")
    if vram_gb >= 10:
        return 8
    elif vram_gb >= 6:
        return 4
    elif vram_gb >= 2:
        return 2
    else:
        return 1


def log_free_memory(device_index: int = 0) -> float:
    """
    Logs available GPU memory and returns it as a float in GB.
    :param device_index: CUDA device index (default = 0)
    :return: Free memory on the selected GPU in GB
    """
    free_mem, total_mem = torch.cuda.mem_get_info(device_index)
    free_mem_gb = free_mem / 1e9
    logger.info(f"GPU free memory: {free_mem_gb:.2f} GB")
    return free_mem_gb