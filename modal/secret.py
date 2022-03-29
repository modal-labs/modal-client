from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_apis

from .object import Object


class _Secret(Object, type_prefix="st"):
    """Secrets provide a dictionary of environment variables for images.

    Secrets are a secure way to add credentials and other sensitive information
    to the containers your functions run in. You can create and edit secrets on
    [the dashboard](https://modal.com/secrets), or programatically from Python code.

    ### Using secrets
    To inject secrets into the container running your function, you add the `secret=` or `secrets=[...]` argument
    to your `modal.function` annotation. For deployed secrets (e.g., secrets defined on the Modal website) you
    can refer to your secrets using `Secret.include(app, secret_name)`:

    ```python
    import modal

    @app.function(secret=modal.Secret.include(app, "my-secret-name"))
    def some_function():
        ...

    @app.function(secrets=[
        modal.Secret.include(app, "my-secret-name"),
        modal.Secret.include(app, "other-secret"),
    ])
    def other_function():
        ...
    ```

    ### Programmatic creation of secrets
    You can programatically create a secret and send it along to your function using `Secret.create`:

    ```python
    import os
    import modal

    @modal.Secret.factory
    def my_local_secret():
        return Secret.create({"FOO": os.environ["LOCAL_FOO"]})

    @app.function(secret=my_local_secret)
    def some_function():
        print(os.environ["FOO"])
    ```

    ### Deploying secrets
    Sometimes, it can be convenient to not have to go through the website to save or update secrets.
    You can then *deploy* secrets similar to how you deploy other objects to Modal, which has the
    same effect as publishing a secret on the web page:

    ```python
    import modal
    app = modal.App()

    if __name__ == "__main__":
        with app.run() as app:
            app.deploy("my-secret-name", Secret.create({"FOO": "BAR"}))
    ```

    The secrets deployed this way will also show up on [the dashboard](https://modal.com/secrets).
    """

    def __init__(self, app, env_dict={}, template_type=""):
        self._env_dict = env_dict
        self._template_type = template_type
        super().__init__(app=app)

    async def load(self, app):
        req = api_pb2.SecretCreateRequest(app_id=app.app_id, env_dict=self._env_dict, template_type=self._template_type)
        resp = await app.client.stub.SecretCreate(req)
        return resp.secret_id


Secret, AioSecret = synchronize_apis(_Secret)
