import subprocess
import re
import sys
import argparse
from typing import Optional
import os


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


def has_staged_py_files() -> bool:
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--cached"],
            capture_output=True,
            text=True,
            check=True
        )
        return any(fname.endswith(".py") for fname in result.stdout.splitlines())
    except subprocess.CalledProcessError:
        return False


def colorize_full_rows(threshold: Optional[int] = None, color_total_line: bool = False, extra_pytest_args=None) -> bool:
    should_run_tests = has_staged_py_files()

    if should_run_tests:
        pytest_cmd = [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--source=src/auto_apply_bot",
            "-m",
            "pytest",
            "tests",
        ]
        if extra_pytest_args:
            pytest_cmd.extend(extra_pytest_args)
        try:
            subprocess.run(pytest_cmd, check=True)
            test_failed = False
        except subprocess.CalledProcessError:
            print("\033[33mTests failed. Coverage will still be shown.\033[0m\n")
            test_failed = True
    else:
        if not os.path.exists(".coverage"):
            print("\033[31mNo staged Python files and no .coverage file found. Exiting.\033[0m")
            return False
        test_failed = False
        print("\033[33mNo staged Python files. Skipping tests and showing existing coverage.\033[0m")

    if threshold is not None:
        print(f"\n{CYAN_BRIGHT}Coverage Report:{f' [TRUNCATED AT < {threshold}%]' if threshold is not None else ''}{RESET}\n")
    else:
        print(f"\n{CYAN_BRIGHT}Coverage Report: {RESET}\n")
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
            _, _, _, cover_str, _ = match.groups()
            percent = int(cover_str.replace('%', ''))
            total_line = percent, line
            continue

        if match:
            _, _, _, cover_str, _ = match.groups()
            percent = int(cover_str.replace('%', ''))
            if threshold is None:   
                rows_with_percent.append((percent, line))
            elif percent < threshold:
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
        if not color_total_line:
            print(f"{MAGENTA_BRIGHT}{line}{RESET}")
        else:
            color = coverage_color(percent)
            print(f"{color}{line}{RESET}")

    return not test_failed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pytest with coverage and colorize results.")
    parser.add_argument("--threshold", type=int, default=None, help="Only show files below this coverage %")
    parser.add_argument("--color-total-line", action="store_true", help="Color the TOTAL row based on its percentage")
    
    args, extra = parser.parse_known_args()

    success = colorize_full_rows(
        threshold=args.threshold,
        color_total_line=args.color_total_line,
        extra_pytest_args=extra
    )
    sys.exit(0 if success else 1)