from .object import Object
from .proto import api_pb2


class Secret(Object, type_prefix="st"):
    """Secrets provide a dictionary of environment variables for images.

    Secrets are a secure way to add credentials and other sensitive information
    to the containers your functions run in. You can create and edit secrets on
    [the dashboard](https://modal.com/secrets), or programatically from Python code.

    ### Using secrets
    To inject secrets into the container running your function, you add the `secret=` or `secrets=[...]` argument
    to your `modal.function` annotation. For deployed secrets (e.g., secrets defined on the Modal website) you
    can refer to your secrets using `Secret.include(secret_name)`:

    ```python
    import modal

    @modal.function(secret=modal.Secret.include("my-secret-name"))
    def some_function():
        ...

    @modal.function(secrets=[
        modal.Secret.include("my-secret-name"),
        modal.Secret.include("other-secret"),
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

    @modal.function(secret=my_local_secret)
    def some_function():
        print(os.environ["FOO"])
    ```

    ### Deploying secrets
    Sometimes, it can be convenient to not have to go through the website to save or update secrets.
    You can then *deploy* secrets similar to how you deploy other objects to Modal, which has the
    same effect as publishing a secret on the web page:

    ```python
    import modal

    if __name__ == "__main__":
        with modal.run() as app:
            app.deploy("my-secret-name", Secret.create({"FOO": "BAR"}))
    ```

    The secrets deployed this way will also show up on [the dashboard](https://modal.com/secrets).
    """

    @classmethod
    async def create(cls, env_dict={}, template_type="", app=None):
        app = cls._get_app(app)
        req = api_pb2.SecretCreateRequest(app_id=app.app_id, env_dict=env_dict, template_type=template_type)
        resp = await app.client.stub.SecretCreate(req)
        return cls._create_object_instance(resp.secret_id, app)
