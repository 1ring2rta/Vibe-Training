from __future__ import annotations

import argparse
import sys
from pathlib import Path

from autopilot.llamafactory.validate import validate_prepared_dataset_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a prepared LLaMA-Factory dataset_dir.")
    parser.add_argument("dataset_dir")
    args = parser.parse_args(argv)
    errors = validate_prepared_dataset_dir(Path(args.dataset_dir))
    if errors:
        for err in errors:
            print(f"[error] {err}", file=sys.stderr)
        return 1
    print(f"[ok] prepared dataset_dir is valid: {Path(args.dataset_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
