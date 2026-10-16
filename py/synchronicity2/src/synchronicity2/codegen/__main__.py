# Copyright Modal Labs 2026
"""Allow running synchronicity2.codegen as a module.

python -m synchronicity2.codegen wrappers -m my.package.impl -o out/
python -m synchronicity2.codegen vendor my.lib.synchronicity2 -o src/
"""

from .cli import main

if __name__ == "__main__":
    main()
