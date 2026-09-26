"""Entry point for ``python -m orion``.

Keeping this a thin shim means the CLI's exit code is the process's exit code,
so ``python -m orion scenario validate`` can gate CI on a non-zero status
without a wrapper script.
"""

from __future__ import annotations

import sys

from orion.cli import main

if __name__ == "__main__":
    sys.exit(main())
