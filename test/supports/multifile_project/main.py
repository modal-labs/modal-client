# Copyright Modal Labs 2024
import a
import b

import modal

app = modal.App()
app.include(a.app)
app.include(b.app)
