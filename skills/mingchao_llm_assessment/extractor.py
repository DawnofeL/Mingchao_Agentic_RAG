"""
Deprecated.

This skill no longer allows slim/intermediate files. Do not use this script in
the assessment workflow. It intentionally exits without writing any files.
"""

import sys


def main() -> int:
    print(
        "extractor.py is deprecated: assessment must read the input JSON in "
        "memory and write only the final _Eval_*.json file.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
