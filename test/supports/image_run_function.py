# Copyright Modal Labs 2022
import modal

app = modal.App("a", include_source=True)  # TODO: remove include_source=True)
other = modal.App("b", include_source=True)  # TODO: remove include_source=True)


def builder_function():
    print("ran builder function")


image = modal.Image.debian_slim().run_function(builder_function)


@app.function(image=image)
def foo():
    pass
