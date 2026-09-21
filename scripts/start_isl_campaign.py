"""Fresh training followed immediately by continuous checkpointed training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--target", type=int, default=12000)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    prefix = f"v7_seed{a.seed}"
    parent = root / "runs" / (prefix + "_block1")
    output = root / "runs" / (prefix + "_continuous")
    state = root / "runs" / (prefix + "_campaign.json")
    if parent.exists() or output.exists() or state.exists():
        p.error("Existing campaign artifacts: refusing overwrite or implicit restart")
    state.parent.mkdir(parents=True, exist_ok=True)

    def write(status, **extra):
        data = dict(status=status, pid=os.getpid(), seed=a.seed,
                    time=time.time(), target=a.target,
                    threads=os.environ.get("M2_TORCH_THREADS", "6"), **extra)
        tmp = state.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        tmp.replace(state)

    try:
        write("bootstrap")
        subprocess.run([sys.executable, "scripts/train_isl.py", "--out", str(parent),
                        "--seed", str(a.seed), "--updates", "100"], cwd=root, check=True)
        write("continuous")
        subprocess.run([sys.executable, "scripts/continue_isl.py", "--parent", str(parent),
                        "--out", str(output), "--seed", str(a.seed),
                        "--target", str(a.target)], cwd=root, check=True)
        write("complete")
    except BaseException as exc:
        write("stopped_error", error=str(exc))
        raise


if __name__ == "__main__":
    main()
