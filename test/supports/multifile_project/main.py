# Copyright Modal Labs 2024
import a
import b

import modal

stub = modal.Stub()
stub.include(a.stub)
stub.include(b.stub)
