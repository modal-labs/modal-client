# Copyright Modal Labs 2022
# See test in client_test/package_utils_test.py
import modal

assert __package__ == "modal_test_support"

stub = modal.Stub("xyz")
