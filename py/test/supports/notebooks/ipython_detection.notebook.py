# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + tags=["parameters"]
server_addr = None
token_id = None
token_secret = None
# -

# + tags=["test_ipython_detection"]
from modal._ipython import is_interactive_ipython

# This should return True when running in a notebook
result = is_interactive_ipython()
print(f"is_interactive_ipython returned: {result}")

# Assert the main test condition
assert result is True
# -
