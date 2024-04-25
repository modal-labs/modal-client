# Copyright Modal Labs 2022
import modal

app = modal.App("a")
other = modal.App("b")


def builder_function():
    print("ran builder function")


image = modal.Image.debian_slim().run_function(builder_function)


@app.function(image=image)
def foo():
    pass
