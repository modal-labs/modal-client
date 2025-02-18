# Copyright Modal Labs 2024
import modal

app = modal.App("class-hierarchy", include_source=True)  # TODO: remove include_source=True)


class Base:
    @modal.method()
    def defined_on_base(self):
        print("base")

    @modal.method()
    def overridden_on_wrapped(self):
        raise NotImplementedError()


@app.cls()
class Wrapped(Base):
    @modal.method()
    def overridden_on_wrapped(self):
        print("wrapped")
