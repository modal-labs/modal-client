# Copyright Modal Labs 2022
import modal

app = modal.App("a", include_source=False)
other = modal.App("b", include_source=False)


def builder_function():
    print("ran builder function")


image = modal.Image.debian_slim().run_function(builder_function, include_source=False)


@app.function(image=image)
def foo():
    pass
