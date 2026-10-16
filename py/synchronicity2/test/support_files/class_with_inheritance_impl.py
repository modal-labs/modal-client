# Copyright Modal Labs 2026
import synchronicity2

m = synchronicity2.Module("class_with_inheritance")


@m.wrap_class()
class WrappedType:
    pass


class UnwrappedBase:
    a: int

    def __init__(self, a: int):
        self.a = a

    def unwrapped_method(self) -> bool:
        return True


@m.wrap_class()
class WrappedBase(UnwrappedBase):
    b: str

    def __init__(self, b: str):
        super().__init__(a=1)
        self.b = "hello"

    async def wrapped_method(self, t: WrappedType) -> list:
        assert isinstance(t, WrappedType), "got wrapped WrappedType instead of unwrapped impl"
        return []


@m.wrap_class()
class WrappedSub(WrappedBase):
    c: float

    def __init__(self, b: str):
        super().__init__(b=b)
        self.c = 1.5

    async def wrapped_in_sub(self) -> dict:
        return {}


@m.wrap_class()
class ClassWithoutOwnMethods(WrappedBase):
    pass
