# Copyright Modal Labs 2022
from modal import App, Image

app = App()

image = Image.debian_slim().add_local_python_source("module_1")


@app.function(image=image, serialized=True)
def num_mounts(_x):
    print(image._mount_layers)
    assert len(image._mount_layers) == 1
    mount = image._mount_layers[0]
    return len(mount.entries)
