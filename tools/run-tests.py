#!/usr/bin/env python3
"""Run nextest and remove only this run's private model directories afterwards.

Triton scans the repository for index requests. Keeping directories until every
parallel test has finished prevents a directory disappearing during that scan.
"""

import os
from pathlib import Path
import shutil
import subprocess
import uuid


def main():
    root = Path(__file__).resolve().parent.parent
    run = uuid.uuid4().hex
    environment = dict(os.environ, TRITON_NG_TEST_RUN=run)
    try:
        return subprocess.call(
            ["cargo", "nextest", "run", "--workspace", "--no-fail-fast"],
            cwd=root,
            env=environment,
        )
    finally:
        for directory in (root / "models").glob(f"test_{run}_*"):
            shutil.rmtree(directory)


if __name__ == "__main__":
    raise SystemExit(main())
