# Copyright Modal Labs 2022
from modal import enter, method


class BaseCls2:
    @enter()
    def enter(self):
        self.x = 2

    @method()
    def run(self, y):
        return self.x * y
