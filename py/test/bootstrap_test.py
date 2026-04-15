# Copyright Modal Labs 2026
import io
import pytest
import zipfile
from unittest import mock

from modal_proto import api_pb2

from .conftest import run_cli_command


def _make_template_zip(repo_name: str, templates: dict[str, dict[str, bytes]]) -> bytes:
    """Create a zip mimicking GitHub's archive format.

    Args:
        repo_name: e.g. "templates" produces root dir "templates-main/"
        templates: mapping of template_name -> {filename: content}
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        root = f"{repo_name}-main/"
        zf.writestr(root, "")
        for template_name, files in templates.items():
            prefix = root + template_name + "/"
            zf.writestr(prefix, "")
            for name, content in files.items():
                zf.writestr(prefix + name, content)
    return buf.getvalue()


def _mock_http_session(zip_bytes: bytes, status: int = 200):
    """Create a mock aiohttp session that returns zip_bytes from any GET."""
    mock_resp = mock.AsyncMock()
    mock_resp.status = status
    mock_resp.read = mock.AsyncMock(return_value=zip_bytes)
    mock_resp.raise_for_status = mock.MagicMock()
    if status >= 400:
        mock_resp.raise_for_status.side_effect = Exception(f"HTTP {status}")

    mock_session = mock.AsyncMock()
    mock_session.get = mock.MagicMock(return_value=mock.AsyncMock(__aenter__=mock.AsyncMock(return_value=mock_resp)))
    mock_session.close = mock.AsyncMock()
    return mock_session


TEMPLATE_ITEMS = [
    api_pb2.TemplateListResponse.TemplateListItem(
        name="hello_world", repo="https://github.com/test-org/templates", ref="main"
    ),
    api_pb2.TemplateListResponse.TemplateListItem(
        name="text_to_image", repo="https://github.com/test-org/templates", ref="main"
    ),
]

TEMPLATE_ZIP = _make_template_zip(
    "templates",
    {
        "hello_world": {
            "app.py": b"# hello world app\nprint('hello')\n",
            "try.py": b"# try it out\n",
        },
        "text_to_image": {
            "app.py": b"# text to image app\n",
            "try.py": b"# try it out\n",
            "subdir/config.yaml": b"model: stable-diffusion\n",
        },
    },
)


@pytest.fixture
def bootstrap_env(servicer, set_env_client, tmp_path, monkeypatch):
    """Common setup for bootstrap tests: templates registered, HTTP mocked, cwd in tmp_path."""
    servicer.template_list_items = TEMPLATE_ITEMS
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("modal.cli.bootstrap._http_client_with_tls", lambda timeout: _mock_http_session(TEMPLATE_ZIP))
    return tmp_path


def test_bootstrap_extracts_template(bootstrap_env):
    res = run_cli_command(["bootstrap", "hello_world"])
    assert "hello_world" in res.stdout
    assert (bootstrap_env / "hello_world" / "app.py").read_text() == "# hello world app\nprint('hello')\n"
    assert (bootstrap_env / "hello_world" / "try.py").read_text() == "# try it out\n"


def test_bootstrap_extracts_subdirectories(bootstrap_env):
    run_cli_command(["bootstrap", "text_to_image"])
    assert (bootstrap_env / "text_to_image" / "app.py").read_text() == "# text to image app\n"
    assert (bootstrap_env / "text_to_image" / "subdir" / "config.yaml").read_text() == "model: stable-diffusion\n"


def test_bootstrap_output_dir(bootstrap_env):
    run_cli_command(["bootstrap", "hello_world", "-o", "my_custom_dir"])
    assert (
        bootstrap_env / "my_custom_dir" / "hello_world" / "app.py"
    ).read_text() == "# hello world app\nprint('hello')\n"
    assert not (bootstrap_env / "hello_world").exists()


def test_bootstrap_cd_step_in_output(bootstrap_env):
    # Default output (cwd) should not suggest a cd step
    res = run_cli_command(["bootstrap", "hello_world"])
    assert "cd " not in res.stdout

    # Custom output dir should suggest a cd step
    res = run_cli_command(["bootstrap", "text_to_image", "-o", "my_project"])
    assert "cd my_project" in res.stdout


def test_bootstrap_existing_dir_fails(bootstrap_env):
    run_cli_command(["bootstrap", "hello_world"])
    run_cli_command(["bootstrap", "hello_world"], expected_exit_code=2, expected_stderr="already exists")


def test_bootstrap_force_overwrites(bootstrap_env):
    run_cli_command(["bootstrap", "hello_world"])
    # Run again with --force to overwrite existing directory
    run_cli_command(["bootstrap", "hello_world", "--force"])
    assert (bootstrap_env / "hello_world" / "app.py").read_text() == "# hello world app\nprint('hello')\n"


def test_bootstrap_unknown_template(bootstrap_env):
    res = run_cli_command(["bootstrap", "nonexistent"], expected_exit_code=1)
    assert "Unknown template: nonexistent" in res.stdout
    assert "hello_world" in res.stdout
    assert "text_to_image" in res.stdout


def test_bootstrap_no_templates(servicer, set_env_client, tmp_path, monkeypatch):
    servicer.template_list_items = []
    monkeypatch.chdir(tmp_path)
    run_cli_command(["bootstrap", "anything"], expected_exit_code=1)


def test_bootstrap_download_failure(bootstrap_env, monkeypatch):
    monkeypatch.setattr(
        "modal.cli.bootstrap._http_client_with_tls", lambda timeout: _mock_http_session(b"", status=500)
    )
    res = run_cli_command(["bootstrap", "hello_world"], expected_exit_code=1)
    assert "Failed to download" in res.stdout


def test_bootstrap_template_not_in_zip(bootstrap_env, monkeypatch):
    wrong_zip = _make_template_zip("templates", {"other_template": {"f.py": b"x"}})
    monkeypatch.setattr("modal.cli.bootstrap._http_client_with_tls", lambda timeout: _mock_http_session(wrong_zip))
    res = run_cli_command(["bootstrap", "hello_world"], expected_exit_code=1)
    assert "not found in repository" in res.stdout


def test_bootstrap_invalid_archive(bootstrap_env, monkeypatch):
    monkeypatch.setattr(
        "modal.cli.bootstrap._http_client_with_tls", lambda timeout: _mock_http_session(b"not a zip at all")
    )
    res = run_cli_command(["bootstrap", "hello_world"], expected_exit_code=1)
    assert "Unable to fetch template" in res.stdout


def test_bootstrap_rejects_zip_path_traversal(servicer, set_env_client, tmp_path, monkeypatch):
    servicer.template_list_items = TEMPLATE_ITEMS
    monkeypatch.chdir(tmp_path)

    # Craft a zip with a path traversal entry
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("templates-main/", "")
        zf.writestr("templates-main/hello_world/", "")
        zf.writestr("templates-main/hello_world/../../etc/malicious", b"pwned")
    traversal_zip = buf.getvalue()

    monkeypatch.setattr("modal.cli.bootstrap._http_client_with_tls", lambda timeout: _mock_http_session(traversal_zip))
    res = run_cli_command(["bootstrap", "hello_world"], expected_exit_code=1)
    assert "Unable to fetch template" in res.stdout
    assert not (tmp_path / "etc" / "malicious").exists()
