# Copyright Modal Labs 2022
import re
from typing import Optional, Union

import rich
import typer
from click import UsageError
from rich.table import Column
from rich.text import Text
from typer import Argument

from modal._object import _get_environment_name
from modal._utils.async_utils import synchronizer
from modal._utils.grpc_utils import get_proto_oneof
from modal.client import _Client
from modal.config import logger
from modal.environments import ensure_env
from modal_proto import api_pb2

from .._utils.time_utils import timestamp_to_localized_str
from .utils import ENV_OPTION, display_table, get_app_id_from_name, stream_app_logs

APP_IDENTIFIER = Argument("", help="App name or ID")
NAME_OPTION = typer.Option("", "-n", "--name", help="Deprecated: Pass App name as a positional argument")

app_cli = typer.Typer(name="app", help="Manage deployed and running apps.", no_args_is_help=True)

APP_STATE_TO_MESSAGE = {
    api_pb2.APP_STATE_DEPLOYED: Text("deployed", style="green"),
    api_pb2.APP_STATE_DETACHED: Text("ephemeral (detached)", style="green"),
    api_pb2.APP_STATE_DETACHED_DISCONNECTED: Text("ephemeral (detached)", style="green"),
    api_pb2.APP_STATE_DISABLED: Text("disabled", style="dim"),
    api_pb2.APP_STATE_EPHEMERAL: Text("ephemeral", style="green"),
    api_pb2.APP_STATE_INITIALIZING: Text("initializing...", style="yellow"),
    api_pb2.APP_STATE_STOPPED: Text("stopped", style="blue"),
    api_pb2.APP_STATE_STOPPING: Text("stopping...", style="blue"),
}


@synchronizer.create_blocking
async def get_app_id(app_identifier: str, env: Optional[str], client: Optional[_Client] = None) -> str:
    """Resolve an app_identifier that may be a name or an ID into an ID."""
    if re.match(r"^ap-[a-zA-Z0-9]{22}$", app_identifier):
        return app_identifier
    return await get_app_id_from_name.aio(app_identifier, env, client)


@app_cli.command("list")
@synchronizer.create_blocking
async def list_(env: Optional[str] = ENV_OPTION, json: bool = False):
    """List Modal apps that are currently deployed/running or recently stopped."""
    env = ensure_env(env)
    client = await _Client.from_env()

    resp: api_pb2.AppListResponse = await client.stub.AppList(
        api_pb2.AppListRequest(environment_name=_get_environment_name(env))
    )

    columns: list[Union[Column, str]] = [
        Column("App ID", min_width=25),  # Ensure that App ID is not truncated in slim terminals
        "Description",
        "State",
        "Tasks",
        "Created at",
        "Stopped at",
    ]
    rows: list[list[Union[Text, str]]] = []
    for app_stats in resp.apps:
        state = APP_STATE_TO_MESSAGE.get(app_stats.state, Text("unknown", style="gray"))
        rows.append(
            [
                app_stats.app_id,
                app_stats.description,
                state,
                str(app_stats.n_running_tasks),
                timestamp_to_localized_str(app_stats.created_at, json),
                timestamp_to_localized_str(app_stats.stopped_at, json),
            ]
        )

    env_part = f" in environment '{env}'" if env else ""
    display_table(columns, rows, json, title=f"Apps{env_part}")


@app_cli.command("logs", no_args_is_help=True)
def logs(
    app_identifier: str = APP_IDENTIFIER,
    *,
    env: Optional[str] = ENV_OPTION,
    timestamps: bool = typer.Option(False, "--timestamps", help="Show timestamps for each log line"),
):
    """Show App logs, streaming while active.

    **Examples:**

    Get the logs based on an app ID:

    ```
    modal app logs ap-123456
    ```

    Get the logs for a currently deployed App based on its name:

    ```
    modal app logs my-app
    ```

    """
    app_id = get_app_id(app_identifier, env)
    stream_app_logs(app_id, show_timestamps=timestamps)


@app_cli.command("rollback", no_args_is_help=True, context_settings={"ignore_unknown_options": True})
@synchronizer.create_blocking
async def rollback(
    app_identifier: str = APP_IDENTIFIER,
    version: str = typer.Argument("", help="Target version for rollback."),
    *,
    env: Optional[str] = ENV_OPTION,
):
    """Redeploy a previous version of an App.

    Note that the App must currently be in a "deployed" state.
    Rollbacks will appear as a new deployment in the App history, although
    the App state will be reset to the state at the time of the previous deployment.

    **Examples:**

    Rollback an App to its previous version:

    ```
    modal app rollback my-app
    ```

    Rollback an App to a specific version:

    ```
    modal app rollback my-app v3
    ```

    Rollback an App using its App ID instead of its name:

    ```
    modal app rollback ap-abcdefghABCDEFGH123456
    ```

    """
    env = ensure_env(env)
    client = await _Client.from_env()
    app_id = await get_app_id.aio(app_identifier, env, client)
    if not version:
        version_number = -1
    else:
        if m := re.match(r"v(\d+)", version):
            version_number = int(m.group(1))
        else:
            raise UsageError(f"Invalid version specifer: {version}")
    req = api_pb2.AppRollbackRequest(app_id=app_id, version=version_number)
    await client.stub.AppRollback(req)
    rich.print("[green]✓[/green] Deployment rollback successful!")


@app_cli.command("stop", no_args_is_help=True)
@synchronizer.create_blocking
async def stop(
    app_identifier: str = APP_IDENTIFIER,
    *,
    env: Optional[str] = ENV_OPTION,
):
    """Stop an app."""
    client = await _Client.from_env()
    app_id = await get_app_id.aio(app_identifier, env)
    req = api_pb2.AppStopRequest(app_id=app_id, source=api_pb2.APP_STOP_SOURCE_CLI)
    await client.stub.AppStop(req)


@app_cli.command("history", no_args_is_help=True)
@synchronizer.create_blocking
async def history(
    app_identifier: str = APP_IDENTIFIER,
    *,
    env: Optional[str] = ENV_OPTION,
    json: bool = False,
):
    """Show App deployment history, for a currently deployed app

    **Examples:**

    Get the history based on an app ID:

    ```
    modal app history ap-123456
    ```

    Get the history for a currently deployed App based on its name:

    ```
    modal app history my-app
    ```

    """
    env = ensure_env(env)
    client = await _Client.from_env()
    app_id = await get_app_id.aio(app_identifier, env, client)
    resp = await client.stub.AppDeploymentHistory(api_pb2.AppDeploymentHistoryRequest(app_id=app_id))

    columns = [
        "Version",
        "Time deployed",
        "Client",
        "Deployed by",
        "Commit",
        "Tag",
    ]
    rows = []
    deployments_with_dirty_commit = False
    for idx, app_stats in enumerate(resp.app_deployment_histories):
        style = "bold green" if idx == 0 else ""

        row = [
            Text(f"v{app_stats.version}", style=style),
            Text(timestamp_to_localized_str(app_stats.deployed_at, json), style=style),
            Text(app_stats.client_version, style=style),
            Text(app_stats.deployed_by, style=style),
        ]

        if app_stats.commit_info.commit_hash:
            short_hash = app_stats.commit_info.commit_hash[:7]
            if app_stats.commit_info.dirty:
                deployments_with_dirty_commit = True
                short_hash = f"{short_hash}*"
            row.append(Text(short_hash, style=style))
        else:
            row.append(None)

        if app_stats.tag:
            row.append(Text(app_stats.tag, style=style))
        else:
            row.append(None)

        rows.append(row)

    # Suppress tag information when no deployments used one
    if not any(row[-1] for row in rows):
        rows = [row[:-1] for row in rows]
        columns = columns[:-1]

    rows = sorted(rows, key=lambda x: int(str(x[0])[1:]), reverse=True)
    display_table(columns, rows, json)

    if deployments_with_dirty_commit and not json:
        rich.print("* - repo had uncommitted changes")


def _proto_type_to_python_type(proto_type: api_pb2.GenericPayloadType) -> str:
    """Convert a protobuf GenericPayloadType to Python type annotation string."""
    if proto_type.base_type == api_pb2.PARAM_TYPE_STRING:
        return "str"
    elif proto_type.base_type == api_pb2.PARAM_TYPE_INT:
        return "int"
    elif proto_type.base_type == api_pb2.PARAM_TYPE_BOOL:
        return "bool"
    elif proto_type.base_type == api_pb2.PARAM_TYPE_BYTES:
        return "bytes"
    elif proto_type.base_type == api_pb2.PARAM_TYPE_NONE:
        return "None"
    elif proto_type.base_type == api_pb2.PARAM_TYPE_LIST:
        if proto_type.sub_types:
            sub_type = _proto_type_to_python_type(proto_type.sub_types[0])
            return f"list[{sub_type}]"
        return "list"
    elif proto_type.base_type == api_pb2.PARAM_TYPE_DICT:
        if len(proto_type.sub_types) >= 2:
            key_type = _proto_type_to_python_type(proto_type.sub_types[0])
            value_type = _proto_type_to_python_type(proto_type.sub_types[1])
            return f"dict[{key_type}, {value_type}]"
        return "dict"
    elif proto_type.base_type == api_pb2.PARAM_TYPE_UNSPECIFIED:
        return "typing.Any"
    elif proto_type.base_type == api_pb2.PARAM_TYPE_UNKNOWN:
        return "typing.Any"
    else:
        return "typing.Any"


def _generate_function_stub(function_name: str, app_name: str, handle_metadata: api_pb2.FunctionHandleMetadata) -> str:
    """Generate Python stub code for a Modal function."""
    lines = []

    # Generate function signature with type annotations
    if handle_metadata.function_schema:
        # Build parameter list with types
        params = []
        for arg in handle_metadata.function_schema.arguments:
            if arg.full_type:
                param_type = _proto_type_to_python_type(arg.full_type)
            else:
                param_type = "typing.Any"

            if arg.has_default:
                params.append(f"{arg.name}: {param_type} = ...")
            else:
                params.append(f"{arg.name}: {param_type}")

        # Return type
        if handle_metadata.function_schema.return_type:
            return_type = _proto_type_to_python_type(handle_metadata.function_schema.return_type)
        else:
            return_type = "typing.Any"

        # Generate a dummy function signature to capture ParamSpec
        param_list = ", ".join(params)

        # Create a dummy function signature
        lines.append(f"def _{function_name}_sig({param_list}) -> {return_type}: ...")
        lines.append("")

        # Use a generic helper to capture the ParamSpec and return properly typed Function
        lines.append(
            f"{function_name} = _with_signature("
            f"_{function_name}_sig, "
            f'modal.Function.from_name("{app_name}", "{function_name}"))'
        )

    else:
        # No schema information available
        lines.append(f'{function_name}: modal.Function = modal.Function.from_name("{app_name}", "{function_name}")')

    return "\n".join(lines)


def _generate_class_stub(class_name: str, app_name: str, handle_metadata) -> str:
    """Generate Python stub code for a Modal class."""
    lines = []
    lines.append(f'{class_name}: modal.Cls = modal.Cls.from_name("{app_name}", "{class_name}")')
    return "\n".join(lines)


@app_cli.command("remote-stub", no_args_is_help=True)
@synchronizer.create_blocking
async def remote_stub(
    app_name: str = typer.Argument(help="Name of the deployed app"),
    *,
    env: Optional[str] = ENV_OPTION,
):
    """Generate Python stub code for remotely accessing functions in a deployed app.

    This command outputs Python source code that creates properly typed Function
    and Cls objects for all the functions and classes in the specified app.

    **Examples:**

    Generate stubs for an app called "my-app":

    ```
    modal app remote-stub my-app
    ```

    Generate stubs for an app in a specific environment:

    ```
    modal app remote-stub my-app --env production
    ```
    """
    env = ensure_env(env)
    client = await _Client.from_env()

    # Get app ID from name
    try:
        app_req = api_pb2.AppGetByDeploymentNameRequest(
            name=app_name,
            environment_name=_get_environment_name(env),
        )
        app_resp = await client.stub.AppGetByDeploymentName(app_req)
        if not app_resp.app_id:
            env_comment = f" in environment '{env}'" if env else ""
            rich.print(f"[red]Error:[/red] Could not find deployed app '{app_name}'{env_comment}")
            raise typer.Exit(1)

        app_id = app_resp.app_id
    except Exception as e:
        rich.print(f"[red]Error:[/red] Failed to find app: {e}")
        raise typer.Exit(1)

    # Get all objects in the app
    try:
        objects_req = api_pb2.AppGetObjectsRequest(
            app_id=app_id,
            include_unindexed=False,
            only_class_function=True,
        )
        objects_resp = await client.stub.AppGetObjects(objects_req)
    except Exception as e:
        rich.print(f"[red]Error:[/red] Failed to get app objects: {e}")
        raise typer.Exit(1)

    # Generate stub code
    stub_lines = [
        "# Generated stub code for Modal app functions and classes",
        f"# App: {app_name}",
        f"# Environment: {env or 'default'}",
        "",
        "import typing",
        "import typing_extensions",
        "import modal",
        "",
        "# Generic helper to capture ParamSpec from function signature",
        "P = typing_extensions.ParamSpec('P')",
        "R = typing.TypeVar('R')",
        "",
        "def _with_signature(sig_func: typing.Callable[P, R], actual_func: modal.Function) -> modal.Function[P, R, R]:",
        '    """Helper to capture ParamSpec from dummy signature and apply it to Function."""',
        "    return actual_func  # type: ignore",
        "",
    ]

    functions_found = []
    classes_found = []

    for item in objects_resp.items:
        obj = item.object
        tag = item.tag

        # Get the handle metadata using the oneof field
        handle_metadata = get_proto_oneof(obj, "handle_metadata_oneof")

        # Debug logging for FunctionHandleMetadata
        if isinstance(handle_metadata, api_pb2.FunctionHandleMetadata):
            logger.debug(f"Function {tag} metadata: {handle_metadata}")
            logger.debug(f"Function {tag} schema: {handle_metadata.function_schema}")
            if handle_metadata.function_schema:
                logger.debug(f"Function {tag} arguments: {list(handle_metadata.function_schema.arguments)}")
                logger.debug(f"Function {tag} return_type: {handle_metadata.function_schema.return_type}")

        if isinstance(handle_metadata, api_pb2.FunctionHandleMetadata):
            if handle_metadata.function_type == api_pb2.Function.FUNCTION_TYPE_FUNCTION:
                # Regular function
                stub_code = _generate_function_stub(tag, app_name, handle_metadata)
                stub_lines.append(stub_code)
                functions_found.append(tag)
            elif handle_metadata.function_type == api_pb2.Function.FUNCTION_TYPE_CLASS:
                # Class service function
                stub_code = _generate_class_stub(tag, app_name, handle_metadata)
                stub_lines.append(stub_code)
                classes_found.append(tag)
        elif isinstance(handle_metadata, api_pb2.ClassHandleMetadata):
            # This shouldn't happen with only_class_function=True, but just in case
            stub_code = _generate_class_stub(tag, app_name, handle_metadata)
            stub_lines.append(stub_code)
            classes_found.append(tag)

    if not functions_found and not classes_found:
        rich.print(f"[yellow]Warning:[/yellow] No functions or classes found in app '{app_name}'")
        return

    # Print the generated stub code
    print("\n".join(stub_lines))

    # Print summary to stderr so it doesn't interfere with the stub output
    import sys

    summary_parts = []
    if functions_found:
        summary_parts.append(f"{len(functions_found)} function(s): {', '.join(functions_found)}")
    if classes_found:
        summary_parts.append(f"{len(classes_found)} class(es): {', '.join(classes_found)}")

    summary = "; ".join(summary_parts)
    print(f"# Found {summary}", file=sys.stderr)
