#!/usr/bin/env python3
import subprocess
import sys
from auto_apply_bot.utils.logger import get_logger


logger = get_logger(__name__, quiet_mode=False, disable_file_logging=True)


def main():
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
    if branch == "main":
        logger.error("You cannot commit directly to 'main'. Please use a feature branch and submit a PR.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
