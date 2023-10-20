# Copyright Modal Labs 2023

import pytest

from modal import forward
from modal.exception import InvalidError


def test_tunnel_outside_container(client):
    with pytest.raises(InvalidError):
        with forward(8000, client=client):
            pass


def test_invalid_port_numbers(container_client):
    for port in (-1, 0, 65536):
        with pytest.raises(InvalidError):
            with forward(port, client=container_client):
                pass


def test_create_tunnel(container_client):
    with forward(8000, client=container_client) as tunnel:
        assert tunnel.host == "8000.modal.test"
        assert tunnel.url == "https://8000.modal.test"
