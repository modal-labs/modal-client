# Copyright Modal Labs 2022


def stop_fetching_inputs():
    """Don't fetch any more inputs from the server, after the current one.
    The container will exit gracefully after the current input is processed."""

    from .app import _container_app

    _container_app.fetching_inputs = False
