# Copyright Modal Labs 2024
import modal

image = modal.Image.debian_slim()
app = modal.App(image=image)


@app.cls()
class ClassWithImage:
    @modal.method()
    def image_is_hydrated(self):
        return image.is_hydrated
