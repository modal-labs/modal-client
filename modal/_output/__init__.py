# Copyright Modal Labs 2025
"""Output sub-package for Modal CLI.

This package contains all output-related functionality including:
- manager.py: OutputManager protocol and disabled implementations
- rich.py: Rich-based output (RichOutputManager, console, progress handlers)
- status.py: FunctionCreationStatus for tracking function creation
- pty.py: PTY utilities and log streaming
"""

# Only re-export the public API
from modal.output import enable_output

__all__ = ["enable_output"]
