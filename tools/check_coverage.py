import subprocess
import sys

def run_coverage():
    print("Running tests with coverage...\n")

    try:
        subprocess.run(
            [sys.executable, "-m", "coverage", "run", "--source=auto_apply_bot", "-m", "pytest", "tests"],
            check=True
        )
    except subprocess.CalledProcessError:
        print("Some tests failed. Showing coverage anyway...\n")

    print("Coverage Report:\n")
    subprocess.run(
        [sys.executable, "-m", "coverage", "report", "--show-missing"]
    )

    print("Done.")

if __name__ == "__main__":
    run_coverage()
