# Copyright Modal Labs 2023
import itertools
import os
import webbrowser
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Tuple

import aiohttp.web
from rich.console import Console

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_api
from modal_utils.http_utils import run_temporary_http_server

from .client import _Client
from .config import _lookup_workspace, _store_user_config, config, config_profiles, user_config_path
from .exception import AuthError


class _TokenFlow:
    def __init__(self, client: _Client):
        self.stub = client.stub

    @asynccontextmanager
    async def start(
        self, utm_source: Optional[str] = None, next_url: Optional[str] = None
    ) -> AsyncGenerator[Tuple[str, str, str], None]:
        """mdmd:hidden"""
        # Run a temporary http server returning the token id on /
        # This helps us add direct validation later
        # TODO(erikbern): handle failure launching server

        async def slash(request):
            headers = {"Access-Control-Allow-Origin": "*"}
            return aiohttp.web.Response(text=self.token_flow_id, headers=headers)

        app = aiohttp.web.Application()
        app.add_routes([aiohttp.web.get("/", slash)])
        async with run_temporary_http_server(app) as url:
            req = api_pb2.TokenFlowCreateRequest(
                utm_source=utm_source,
                next_url=next_url,
                localhost_port=int(url.split(":")[-1]),
            )
            resp = await self.stub.TokenFlowCreate(req)
            self.token_flow_id = resp.token_flow_id
            self.wait_secret = resp.wait_secret
            yield (resp.token_flow_id, resp.web_url, resp.code)

    async def finish(
        self, timeout: float = 40.0, grpc_extra_timeout: float = 5.0
    ) -> Optional[api_pb2.TokenFlowWaitResponse]:
        """mdmd:hidden"""
        # Wait for token flow to finish
        req = api_pb2.TokenFlowWaitRequest(
            token_flow_id=self.token_flow_id, timeout=timeout, wait_secret=self.wait_secret
        )
        resp = await self.stub.TokenFlowWait(req, timeout=(timeout + grpc_extra_timeout))
        if not resp.timeout:
            return resp
        else:
            return None


TokenFlow = synchronize_api(_TokenFlow)


async def _new_token(
    *,
    profile: Optional[str] = None,
    activate: bool = True,
    verify: bool = True,
    source: Optional[str] = None,
    next_url: Optional[str] = None,
):
    server_url = config.get("server_url", profile=profile)

    console = Console()

    result: Optional[api_pb2.TokenFlowWaitResponse] = None
    client = await _Client.unauthenticated_client(server_url)
    async with client:
        token_flow = _TokenFlow(client)

        async with token_flow.start(source, next_url) as (_, web_url, code):
            with console.status("Waiting for authentication in the web browser", spinner="dots"):
                # Open the web url in the browser
                if _open_url(web_url):
                    console.print(
                        "The web browser should have opened for you to authenticate and get an API token.\n"
                        "If it didn't, please copy this URL into your web browser manually:\n"
                    )
                else:
                    console.print(
                        "[red]Was not able to launch web browser[/red]\n"
                        "Please go to this URL manually and complete the flow:\n"
                    )
                console.print(f"[link={web_url}]{web_url}[/link]\n")
                if code:
                    console.print(f"Enter this code: [yellow]{code}[/yellow]\n")

            with console.status("Waiting for token flow to complete...", spinner="dots") as status:
                for attempt in itertools.count():
                    result = await token_flow.finish()
                    if result is not None:
                        break
                    status.update(f"Waiting for token flow to complete... (attempt {attempt+2})")

        console.print("[green]Web authentication finished successfully![/green]")

    assert result is not None

    if result.workspace_username:
        console.print(
            f"[green]Token is connected to the [magenta]{result.workspace_username}[/magenta] workspace.[/green]"
        )

    await _set_token(result.token_id, result.token_secret, profile=profile, activate=activate, verify=verify)


async def _set_token(
    token_id: str,
    token_secret: str,
    *,
    profile: Optional[str] = None,
    activate: bool = True,
    verify: bool = True,
):
    # TODO add server_url as a parameter for verification?
    server_url = config.get("server_url", profile=profile)
    console = Console()
    if verify:
        console.print(f"Verifying token against [blue]{server_url}[/blue]")
        await _Client.verify(server_url, (token_id, token_secret))
        console.print("[green]Token verified successfully![/green]")

    if profile is None:
        if "MODAL_PROFILE" in os.environ:
            profile = os.environ["MODAL_PROFILE"]
        else:
            try:
                workspace = await _lookup_workspace(server_url, token_id, token_secret)
            except AuthError as exc:
                if not verify:
                    # Improve the error message for verification failure with --no-verify to reduce surprise
                    msg = "No profile name given, but could not authenticate client to look up workspace name."
                    raise AuthError(msg) from exc
                raise exc
            profile = workspace.username

    config_data = {"token_id": token_id, "token_secret": token_secret}
    # Activate the profile when requested or if no other profiles currently exist
    active_profile = profile if (activate or not config_profiles()) else None
    with console.status("Storing token", spinner="dots"):
        _store_user_config(config_data, profile=profile, active_profile=active_profile)
    console.print(
        f"[green]Token written to [magenta]{user_config_path}[/magenta] in profile [magenta]{profile}[/magenta].[/green]"
    )


def _open_url(url: str) -> bool:
    """Opens url in web browser, making sure we use a modern one (not Lynx etc)"""
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False
    try:
        browser = webbrowser.get()
        # zpresto defines `BROWSER=open` by default on macOS, which causes `webbrowser` to return `GenericBrowser`.
        if isinstance(browser, webbrowser.GenericBrowser) and browser.name != "open":
            return False
        else:
            return browser.open_new_tab(url)
    except webbrowser.Error:
        return False
