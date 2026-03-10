# Copyright Modal Labs 2025
"""Entry point wrapper for gen_cli_docs as a standalone binary."""

import sys

from modal_docs.gen_cli_docs import run

run(None if len(sys.argv) <= 1 else sys.argv[1])
