import logging
from datetime import datetime
from pathlib import Path
from auto_apply_bot import resolve_project_source

def get_logger(name: str) -> logging.Logger:
    proj_root = resolve_project_source()
    logs_dir = proj_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = logs_dir / f"{name.replace('.', '_')}_{timestamp}.log"

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        file_handler = logging.FileHandler(log_file, mode="w")
        file_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                                           datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file {log_file}")
    return logger