# Copyright Modal Labs 2024
import sys

import modal  # noqa

assert "modal" in sys.modules

# This is a very heavy dependency that takes about 70-80ms to import
# Let's make sure it doesn't get imported in global scope
assert "aiohttp" not in sys.modules
