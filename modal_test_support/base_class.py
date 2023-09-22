# Copyright Modal Labs 2022
from modal import method


class BaseCls2:
    def __enter__(self):
        self.x = 2

    @method()
    def run(self, y):
        return self.x * y
