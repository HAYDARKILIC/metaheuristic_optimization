"""Execute every ```python block in solutions/*.md as a standalone script.

Each block is run from inside its week folder (so ``sys.path.insert(0, "..")``
finds ``utils/``) with a non-interactive matplotlib backend. Blocks must be
self-contained: their own imports and helper definitions, no state from the
notebook or from earlier blocks.

Usage:
    python tools/run_solution_blocks.py                 # all weeks
    python tools/run_solution_blocks.py week2_simulated_annealing
"""

import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRELUDE = "import matplotlib\nmatplotlib.use('Agg')\n"
TIMEOUT = 300


def blocks(md_path):
    return re.findall(r"```python\n(.*?)```", md_path.read_text(encoding="utf-8"), flags=re.S)


def run_week(md_path):
    week = md_path.stem
    cwd = ROOT / week
    failures = []
    code_blocks = blocks(md_path)
    for i, code in enumerate(code_blocks):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(PRELUDE + code)
            script = fh.name
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, script], cwd=cwd, capture_output=True, text=True, timeout=TIMEOUT)
            ok, msg = r.returncode == 0, (r.stderr.strip().splitlines() or ["?"])[-1]
        except subprocess.TimeoutExpired:
            ok, msg = False, f"timeout after {TIMEOUT}s"
        status = "ok  " if ok else "FAIL"
        print(f"  [{status}] block {i:02d} ({time.time() - t0:5.1f}s){'' if ok else ' ' + msg[:150]}")
        if not ok:
            failures.append(i)
    return len(code_blocks), failures


def main():
    wanted = set(sys.argv[1:])
    total, failed = 0, 0
    for md in sorted((ROOT / "solutions").glob("week*.md")):
        if wanted and md.stem not in wanted:
            continue
        print(md.name)
        n, fails = run_week(md)
        total += n
        failed += len(fails)
    print(f"\n{total - failed}/{total} solution code blocks ran successfully")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
