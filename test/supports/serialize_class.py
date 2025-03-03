# Copyright Modal Labs 2024
import sys

import modal
from modal import enter, fastapi_endpoint, method
from modal._serialization import serialize


class UserCls:
    @enter()
    def enter(self):
        pass

    @method()
    def method(self):
        return "a"

    @fastapi_endpoint()
    def web_endpoint(self):
        pass


app = modal.App()
app.cls()(UserCls)  # avoid warnings about not turning methods into functions

sys.stdout.buffer.write(serialize(UserCls))
