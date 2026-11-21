# Copyright Modal Labs 2025
import pytest
from datetime import datetime, timezone
from typing import Any

import modal
import modal.experimental
from modal.exception import DeprecationError, InvalidError, NotFoundError
from modal_proto import api_pb2

app = modal.App(include_source=False)


@app.function()
def f():
    pass


@app.cls()
class C:
    @modal.method()
    def method(self):
        pass


@app.server(port=8000, routing_region="us-east", serialized=True)
class SimpleServer:
    @modal.enter()
    def start(self):
        pass


def test_server_experimental_list_containers(client, servicer):
    app.deploy(name="test", client=client)
    function_id = servicer.app_objects[app.app_id]["SimpleServer"]
    response_container = api_pb2.FlashContainerListResponse.Container(
        task_id="ta-123",
        host="server.example.com",
        port=443,
    )
    server = modal.Server.from_name("test", "SimpleServer", client=client)

    with servicer.intercept() as ctx:
        ctx.add_response(
            "FlashContainerList",
            api_pb2.FlashContainerListResponse(containers=[response_container]),
        )
        containers = server._experimental_list_containers()

    assert containers == [
        modal.types.ServerContainerInfo(
            container_id="ta-123",
            host="server.example.com",
            port=443,
        )
    ]
    assert ctx.pop_request("FlashContainerList").function_id == function_id


def test_app_get_objects(client, servicer):
    app.deploy(name="test", environment_name="dev", client=client)
    res = modal.experimental.get_app_objects("test", environment_name="dev", client=client)
    assert len(res) == 3
    assert res.keys() == {"C", "SimpleServer", "f"}
    assert isinstance(res["C"], modal.Cls)
    assert isinstance(res["f"], modal.Function)
    assert isinstance(res["SimpleServer"], modal.Function)


def test_image_delete(client, servicer):
    with app.run(client=client):
        image = modal.Image.debian_slim().build(app)

    assert image.object_id in servicer.images
    modal.experimental.image_delete(image.object_id, client=client)
    assert image.object_id not in servicer.images

    with pytest.raises(NotFoundError):
        modal.experimental.image_delete("im-nonexistent", client=client)


def test_stop_app(client, servicer):
    app_name, environment_name = "test", "dev"
    app.deploy(name=app_name, environment_name=environment_name, client=client)
    with pytest.warns(DeprecationError, match="App.lookup"):
        modal.experimental.stop_app(app_name, environment_name=environment_name, client=client)
    with pytest.raises(NotFoundError):
        modal.App.lookup(app_name, environment_name=environment_name, client=client)


def test_get_app_lifecycle(client, servicer):
    with servicer.intercept() as ctx:
        ctx.add_response(
            "AppGetLifecycle",
            api_pb2.AppGetLifecycleResponse(
                lifecycle=api_pb2.AppLifecycle(
                    created_at=1000.0,
                    created_by="alice",
                    deployed_at=2000.0,
                    deployed_by="bob",
                    version=3,
                    stopped_at=3000.0,
                    stopped_by="carol",
                )
            ),
        )
        lifecycle = modal.experimental.get_app_lifecycle("ap-123", client=client)

    assert ctx.pop_request("AppGetLifecycle").app_id == "ap-123"
    assert lifecycle.created_at == datetime.fromtimestamp(1000.0, timezone.utc)
    assert lifecycle.created_by == "alice"
    assert lifecycle.deployed_at == datetime.fromtimestamp(2000.0, timezone.utc)
    assert lifecycle.deployed_by == "bob"
    assert lifecycle.stopped_at == datetime.fromtimestamp(3000.0, timezone.utc)
    assert lifecycle.stopped_by == "carol"


def test_get_app_lifecycle_running(client, servicer):
    # A running App has no stop event, and an ephemeral App was never deployed; the server
    # returns 0 timestamps and empty strings, which should surface as None.
    with servicer.intercept() as ctx:
        ctx.add_response(
            "AppGetLifecycle",
            api_pb2.AppGetLifecycleResponse(lifecycle=api_pb2.AppLifecycle(created_at=1000.0, created_by="alice")),
        )
        lifecycle = modal.experimental.get_app_lifecycle("ap-123", client=client)

    assert lifecycle.created_at == datetime.fromtimestamp(1000.0, timezone.utc)
    assert lifecycle.deployed_at is None
    assert lifecycle.deployed_by is None
    assert lifecycle.stopped_at is None
    assert lifecycle.stopped_by is None


def test_list_cluster_reservations(client, servicer):
    with servicer.intercept() as ctx:
        ctx.add_response(
            "ClusterReservationList",
            api_pb2.ClusterReservationListResponse(
                reservations=[
                    api_pb2.ClusterReservation(
                        name="b300s austin",
                        gpu_type="B300",
                        region="austin",
                        min_reserved_workers=64,
                        reserved_workers=62,
                        active_workers=61,
                        idle_workers=1,
                        unfulfilled_workers=2,
                        tasks=[
                            api_pb2.ClusterReservation.Task(
                                task_id="ta-1",
                                cluster_rank=0,
                                app_id="ap-1",
                                function_id="fu-1",
                            )
                        ],
                    )
                ]
            ),
        )
        reservations = modal.experimental.list_cluster_reservations(client=client)

    assert ctx.pop_request("ClusterReservationList") is not None
    assert len(reservations) == 1
    reservation = reservations[0]
    assert isinstance(reservation, modal.experimental.ClusterReservation)
    assert reservation.name == "b300s austin"
    assert reservation.gpu_type == "B300"
    assert reservation.min_reserved_workers == 64
    assert reservation.reserved_workers == 62
    assert reservation.active_workers == 61
    assert reservation.idle_workers == 1
    assert reservation.unfulfilled_workers == 2
    assert reservation.tasks == [
        modal.experimental.ClusterReservationTask(task_id="ta-1", cluster_rank=0, app_id="ap-1", function_id="fu-1")
    ]


def test_list_cluster_reservations_empty(client, servicer):
    with servicer.intercept() as ctx:
        ctx.add_response("ClusterReservationList", api_pb2.ClusterReservationListResponse())
        assert modal.experimental.list_cluster_reservations(client=client) == []


def _localized_app(kind, localized_options, **options):
    app = modal.App(include_source=False)
    decorator = modal.experimental.localized(**localized_options)
    if kind == "function":
        app.function(serialized=True, **options)(decorator(lambda: None))
    elif kind == "cls":

        @app.cls(serialized=True, **options)
        @decorator
        class Service:
            @modal.method()
            def run(self):
                pass
    else:

        @app.server(serialized=True, **options)
        @decorator
        class Server:
            @modal.enter()
            def start(self):
                pass

    return app


@pytest.mark.parametrize("kind", ["function", "cls", "server"])
@pytest.mark.parametrize("routing_region", [None, "us-west"])
@pytest.mark.parametrize("target_location", [None, "san-jose"])
def test_localized_routing(client, servicer, kind, routing_region, target_location):
    options = {} if routing_region is None else {"routing_region": routing_region}
    app = _localized_app(kind, {"target_location": target_location}, **options)
    with servicer.intercept() as ctx, app.run(client=client):
        request = ctx.get_requests("FunctionCreate")[0]
        definition = request.function
        assert definition.HasField("localized_config")
        assert definition.localized_config == api_pb2.LocalizedConfig(target_location=target_location)
        if kind == "server":
            assert list(definition.http_config.proxy_regions) == [routing_region or "us-east"]
        else:
            assert definition.routing_region == (routing_region or "")


@pytest.mark.parametrize("kind", ["function", "cls", "server"])
@pytest.mark.parametrize(
    "gpu,localized_options,expected",
    [
        (
            None,
            {"max_distance": "short"},
            {"target_location": None, "preferred_distance": None, "max_distance": "short"},
        ),
        (
            "H100",
            {"preferred_distance": "short"},
            {"target_location": None, "preferred_distance": "short", "max_distance": None},
        ),
        (
            "H100",
            {"target_location": "ashburn", "preferred_distance": "medium", "max_distance": "long"},
            {"target_location": "ashburn", "preferred_distance": "medium", "max_distance": "long"},
        ),
        (
            ["H100", "A100"],
            {"target_location": "ashburn", "preferred_distance": "medium", "max_distance": "long"},
            {"target_location": "ashburn", "preferred_distance": "medium", "max_distance": "long"},
        ),
    ],
)
def test_localized_options(client, servicer, kind, gpu, localized_options, expected):
    experimental_options = {"other_option": True}
    region_option = "compute_region" if kind == "server" else "region"
    app = _localized_app(
        kind, localized_options, gpu=gpu, experimental_options=experimental_options, **{region_option: "us"}
    )
    with servicer.intercept() as ctx, app.run(client=client):
        request = ctx.get_requests("FunctionCreate")[0]
        if isinstance(gpu, list):
            assert request.HasField("function_data")
            assert len(request.function_data.ranked_functions) == 2
            assert request.function_data.localized_config == api_pb2.LocalizedConfig(**expected)
            definitions = [ranked.function for ranked in request.function_data.ranked_functions]
        else:
            definitions = [request.function]
        for definition in definitions:
            assert definition.localized_config == api_pb2.LocalizedConfig(**expected)
            assert definition.experimental_options["other_option"] == "True"
            assert list(definition.scheduler_placement.regions) == ["us"]
    assert experimental_options == {"other_option": True}


@pytest.mark.parametrize("target_location", [None, "frankfurt"])
def test_localized_routing_override(client, servicer, target_location):
    app = modal.App(include_source=False)

    @app.function(serialized=True, routing_region="us-east")
    @modal.experimental.localized(target_location=target_location)
    def f():
        pass

    with servicer.intercept() as ctx, app.run(client=client):
        f.with_options(routing_region="us-west").hydrate()
        definition = ctx.get_requests("FunctionCreate")[0].function
        assert definition.localized_config == api_pb2.LocalizedConfig(target_location=target_location)
        assert ctx.get_requests("FunctionBindParams")[0].function_options.routing_region == "us-west"


@pytest.mark.parametrize("kind", ["function", "cls", "server"])
def test_localized_legacy_target_conflict(kind):
    with pytest.raises(InvalidError, match="cannot be combined with the `target_location` experimental option"):
        _localized_app(kind, {}, experimental_options={"target_location": "ashburn"})


@pytest.mark.parametrize("target_location", [None, "ashburn"])
def test_localized_requires_opt_in(client, servicer, target_location):
    app = modal.App(include_source=False)
    options = {} if target_location is None else {"target_location": target_location}
    app.function(serialized=True, routing_region="us-west", experimental_options=options)(lambda: None)
    with app.run(client=client):
        definition = next(iter(servicer.app_functions.values()))
        assert dict(definition.experimental_options) == options
        assert not definition.HasField("localized_config")


@pytest.mark.parametrize(
    "parameter,value",
    [("target_location", ("ashburn", "short")), ("preferred_distance", False), ("max_distance", {"distance": "short"})],
)
def test_localized_argument_types(parameter, value):
    with pytest.raises(InvalidError, match=f"`{parameter}` must be a string or None"):
        modal.experimental.localized(**{parameter: value})


def test_localized_forwards_location_and_distance_names(client, servicer):
    options = {"target_location": "new-city", "preferred_distance": "new-category", "max_distance": "short"}
    app = _localized_app("function", options)
    with servicer.intercept() as ctx, app.run(client=client):
        assert ctx.get_requests("FunctionCreate")[0].function.localized_config == api_pb2.LocalizedConfig(**options)


@pytest.mark.parametrize("localized_first", [False, True])
def test_localized_clustered_rejected(localized_first):
    decorators = [modal.experimental.localized(), modal.clustered(size=2)]
    if localized_first:
        decorators.reverse()
    with pytest.raises(InvalidError, match="not supported for clustered"):
        decorators[0](decorators[1](lambda: None))


def test_localized_duplicate():
    with pytest.raises(InvalidError, match="only be applied once"):
        modal.experimental.localized()(modal.experimental.localized()(lambda: None))


def test_localized_wrong_order():
    app = modal.App(include_source=False)
    with pytest.raises(InvalidError, match="swap the order"):
        modal.experimental.localized()(app.function(serialized=True)(lambda: None))  # type: ignore[call-overload]


@pytest.mark.parametrize("method_decorator", [modal.method, modal.enter])
@pytest.mark.parametrize("localized_first", [False, True])
def test_localized_method_rejected(method_decorator, localized_first):
    app = modal.App(include_source=False)
    decorators: list[Any] = [modal.experimental.localized(), method_decorator()]
    if localized_first:
        decorators.reverse()
    with pytest.raises(InvalidError, match="to the class, not an individual method"):

        @app.cls(serialized=True)
        class Service:
            @decorators[0]
            @decorators[1]
            def run(self):
                pass


@pytest.mark.parametrize("localized_first", [False, True])
def test_localized_concurrent_web_function(client, servicer, localized_first):
    decorators: list[Any] = [modal.experimental.localized(), modal.concurrent(max_inputs=10)]
    if localized_first:
        decorators.reverse()
    app = modal.App(include_source=False)
    wrapped = decorators[0](decorators[1](modal.fastapi_endpoint()(lambda: None)))
    app.function(serialized=True)(wrapped)
    with app.run(client=client):
        definition = next(iter(servicer.app_functions.values()))
        assert definition.HasField("localized_config")
        assert definition.max_concurrent_inputs == 10
        assert definition.webhook_config.type == api_pb2.WEBHOOK_TYPE_FUNCTION
