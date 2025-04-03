import sys
import subprocess
from pathlib import Path

def should_run_tests(files) -> bool:
    return any(Path(f).suffix == ".py" for f in files)

if __name__ == "__main__":
    filenames = sys.argv[1:]

    if should_run_tests(filenames):
        print("Running tests and generating coverage...\n")
        cmd = [
            sys.executable,
            "-m", "coverage", "run",
            "--source=src/auto_apply_bot",
            "-m", "pytest",
            "tests",
            "--quiet-logs",
            "-vv",
        ]
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    else:
        print("No Python files staged — skipping test run.")
        sys.exit(0)
