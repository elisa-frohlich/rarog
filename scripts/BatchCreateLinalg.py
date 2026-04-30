import os
import subprocess
import sys
import multiprocessing
from contextlib import contextmanager
from tqdm import tqdm

# -----------------------
# Configuration
# -----------------------
CUR_DIR = os.getcwd()
os.chdir("/".join(sys.argv[0].split("/")[:-1]) + "/..")
os.chdir(CUR_DIR)

# Limit for test run (Set to None for full run)
TEST_LIMIT: int | None = None


# -----------------------
# Helpers
# -----------------------
@contextmanager
def suppress_output():
    """Redirects C++ and Python logs to /dev/null."""
    with open(os.devnull, "w") as devnull:
        old_stdout = os.dup(sys.stdout.fileno())
        old_stderr = os.dup(sys.stderr.fileno())
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(old_stdout, sys.stdout.fileno())
            os.dup2(old_stderr, sys.stderr.fileno())
            os.close(old_stdout)
            os.close(old_stderr)


# -----------------------
# Worker Function
# -----------------------
def create_linalg_script(idx: int) -> bool:
    try:
        subprocess.check_call(
            ["./scripts/create_linalg.sh"], env=dict(os.environ, MODEL_IDX=str(idx))
        )
    except subprocess.CalledProcessError:
        # There was an error - command exited with non-zero code
        return False

    return True


# -----------------------
# Main
# -----------------------
def main():
    last_model_idx: int = TEST_LIMIT if TEST_LIMIT else 423624

    tasks = list(range(1, last_model_idx + 1))

    if TEST_LIMIT:
        print(f"DEBUG: Limiting run to first {TEST_LIMIT} models.")

    max_workers = os.cpu_count() or 1
    num_workers = max(1, max_workers - 2)

    print(f"Starting parallel conversion with {num_workers} workers.")

    errors = 0

    with multiprocessing.Pool(processes=num_workers) as pool:
        for success in tqdm(
            pool.imap_unordered(create_linalg_script, tasks, chunksize=50),
            total=len(tasks),
            unit="model",
        ):
            if not success:
                errors += 1

    if errors:
        print(f"\nFinished with {errors} errors.")
    else:
        print("\nFinished successfully with 0 errors.")


if __name__ == "__main__":
    main()
