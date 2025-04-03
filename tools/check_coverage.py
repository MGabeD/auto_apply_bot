import subprocess
import re
import sys

GRADIENT = [
    "\033[91m",    # 0-9% red
    "\033[91m",    # 10-19%
    "\033[91m",    # 20-29%
    "\033[91m",    # 30-39%
    "\033[93m",    # 40-49% yellow
    "\033[93m",    # 50-59%
    "\033[92m",    # 60-69%
    "\033[92m",    # 70-79%
    "\033[92m",    # 80-89%
    "\033[92m",    # 90-99%
    "\033[1;92m",  # 100% bold green
]
RESET = "\033[0m"

def coverage_color(percentage):
    index = min(int(percentage // 10), 10)
    return GRADIENT[index]

def colorize_full_rows(threshold=None):
    try:
        subprocess.run(
            [sys.executable, "-m", "coverage", "run", "--source=src/auto_apply_bot", "-m", "pytest", "tests"],
            check=True
        )
    except subprocess.CalledProcessError:
        print("\033[33mTests failed. Coverage will still be shown.\033[0m\n")

    print("\n\033[36mCoverage Report:\033[0m\n")
    result = subprocess.run(["coverage", "report", "--show-missing"], capture_output=True, text=True)
    lines = result.stdout.splitlines()

    for line in lines:
        match = re.match(r"^(.*?)\s+(\d+)\s+(\d+)\s+(\d+%)\s*(.*)$", line)
        if match:
            _, _, _, cover_str, _ = match.groups()
            percent = int(cover_str.replace('%', ''))
            if threshold is not None and percent >= threshold:
                continue
            color = coverage_color(percent)
            print(f"{color}{line}{RESET}")
        else:
            print(line)

    print(f"\n\033[32mDone.{RESET}")

if __name__ == "__main__":
    colorize_full_rows(threshold=100)