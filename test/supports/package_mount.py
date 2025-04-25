# Copyright Modal Labs 2022
from modal import App, Image

app = App(include_source=True)  # TODO: remove include_source=True when automount is disabled by default

# just make sure that non-existing package doesn't cause this to crash in containers:
image = Image.debian_slim().add_local_python_source("non_existing_package_123154")


@app.function(image=image, serialized=True)
def dummy(_x):
    return 0
