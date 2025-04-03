import subprocess
import re
import sys
import argparse
from typing import Optional
from pathlib import Path

# Rich 256-color gradient (red → yellow-heavy → green)
GRADIENT = [
    "\033[38;5;196m",  # 0-9%
    "\033[38;5;202m",  # 10-19%
    "\033[38;5;208m",  # 20-29%
    "\033[38;5;214m",  # 30-39%
    "\033[38;5;220m",  # 40-49%
    "\033[38;5;226m",  # 50-59%
    "\033[38;5;190m",  # 60-69%
    "\033[38;5;154m",  # 70-79%
    "\033[38;5;118m",  # 80-89%
    "\033[38;5;82m",   # 90-99%
    "\033[38;5;46m",   # 100%
]

CYAN_BRIGHT = "\033[38;5;45m"
MAGENTA_BRIGHT = "\033[38;5;201m"
RESET = "\033[0m"

def coverage_color(percentage):
    index = min(int(percentage // 10), 10)
    return GRADIENT[index]

def colorize_coverage_report(threshold: Optional[int] = None, color_total_line: bool = False) -> bool:
    if not Path(".coverage").exists():
        print(f"{MAGENTA_BRIGHT}No .coverage file found. Please run `coverage run` before this script.{RESET}")
        return False

    print(f"\n{CYAN_BRIGHT}Coverage Report:{' [TRUNCATED at: {threshold}%]' if threshold is not None else ''}{RESET}\n")
    
    result = subprocess.run(["coverage", "report", "--show-missing"], capture_output=True, text=True)
    lines = result.stdout.splitlines()

    header_lines = lines[:2]
    data_lines = lines[2:]

    rows_with_percent = []
    total_line = None
    other_nonmatch_lines = []

    for line in data_lines:
        match = re.match(r"^(.*?)\s+(\d+)\s+(\d+)\s+(\d+%)\s*(.*)$", line)

        if line.strip().startswith("TOTAL"):
            if match:
                _, _, _, cover_str, _ = match.groups()
                percent = int(cover_str.replace('%', ''))
                total_line = percent, line
            continue

        if match:
            _, _, _, cover_str, _ = match.groups()
            percent = int(cover_str.replace('%', ''))
            if threshold is None or percent < threshold:
                rows_with_percent.append((percent, line))
        else:
            other_nonmatch_lines.append(line)

    rows_with_percent.sort(key=lambda x: x[0], reverse=True)

    for line in header_lines:
        print(f"{MAGENTA_BRIGHT}{line}{RESET}")

    for percent, line in rows_with_percent:
        color = coverage_color(percent)
        print(f"{color}{line}{RESET}")

    for line in other_nonmatch_lines:
        print(f"{MAGENTA_BRIGHT}{line}{RESET}")

    if total_line:
        percent, line = total_line
        color = coverage_color(percent) if color_total_line else MAGENTA_BRIGHT
        print(f"{color}{line}{RESET}")

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colorize existing coverage report.")
    parser.add_argument("--threshold", type=int, default=None, help="Only show files below this coverage %")
    parser.add_argument("--color-total-line", action="store_true", help="Color the TOTAL row based on its percentage")
    args = parser.parse_args()

    success = colorize_coverage_report(threshold=args.threshold, color_total_line=args.color_total_line)
    sys.exit(0 if success else 1)
