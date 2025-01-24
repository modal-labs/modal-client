# Copyright Modal Labs 2022
from modal import App, Image

app = App()

# just make sure that non-existing package doesn't cause this to crash in containers:
image = Image.debian_slim().add_local_python_source("non_existing_package_123154")


@app.function(image=image, serialized=True)
def dummy(_x):
    return 0
