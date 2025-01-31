# Copyright Modal Labs 2025
from modal import App, Image, Queue, Volume

app = App()

image = Image.debian_slim().pip_install("xyz")
volume = Volume.from_name("my-vol")
queue = Queue.from_name("my-queue")


@app.function(image=image, volumes={"/tmp/xyz": volume})
def f(x):
    # These are hydrated by virtue of being dependencies
    assert image.is_hydrated
    assert volume.is_hydrated

    # This one should be hydrated lazily
    queue.put("123")
    assert queue.get() == "123"
