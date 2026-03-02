# Copyright Modal Labs 2026
import json
from unittest import mock

from .conftest import run_cli_command

SAMPLE_CHANGELOG = """\
# Changelog

## Latest

### 1.3.3 (2026-02-12)

- Fixed a bug in volume mounts.
- Improved error messages for GPU selection.

### 1.3.2 (2026-02-05)

- Added support for new instance types.

### 1.3.0 (2026-01-20)

- Initial 1.3 release with new features.
- New `modal changelog` command.

## 1.2

### 1.2.5 (2026-01-10)

- Backport fix for container cleanup.

### 1.2.4 (2026-01-05)

- Minor improvements to sandbox API.

### 1.2.0 (2025-12-15)

- Major 1.2 release.
"""


def _mock_urlopen(markdown=SAMPLE_CHANGELOG):
    resp = mock.MagicMock()
    resp.read.return_value = markdown.encode("utf-8")
    resp.__enter__ = mock.Mock(return_value=resp)
    resp.__exit__ = mock.Mock(return_value=False)
    return mock.patch("modal.cli.changelog.urllib.request.urlopen", return_value=resp)


def test_changelog_default(set_env_client):
    with _mock_urlopen(), mock.patch("modal_version.__version__", "1.3.3"):
        res = run_cli_command(["changelog"])
    assert "### 1.3.3" in res.stdout
    assert "### 1.3.2" in res.stdout
    assert "### 1.3.0" in res.stdout
    assert "### 1.2" not in res.stdout


def test_changelog_last_n(set_env_client):
    # With version 1.3.2, --last 2 should show 1.3.2 and 1.3.0 (skipping 1.3.3 which is newer)
    with _mock_urlopen(), mock.patch("modal_version.__version__", "1.3.2"):
        res = run_cli_command(["changelog", "--last", "2"])
    assert "### 1.3.3" not in res.stdout
    assert "### 1.3.2" in res.stdout
    assert "### 1.3.0" in res.stdout
    assert "### 1.2" not in res.stdout


def test_changelog_since(set_env_client):
    with _mock_urlopen():
        res = run_cli_command(["changelog", "--since", "1.2.5"])
    assert "### 1.3.3" in res.stdout
    assert "### 1.3.0" in res.stdout
    assert "### 1.2.5" not in res.stdout
    assert "### 1.2.4" not in res.stdout


def test_changelog_since_date(set_env_client):
    with _mock_urlopen():
        res = run_cli_command(["changelog", "--since", "2026-01-20"])
    assert "### 1.3.3" in res.stdout
    assert "### 1.3.2" in res.stdout
    assert "### 1.3.0" not in res.stdout  # 2026-01-20 is not > 2026-01-20
    assert "### 1.2" not in res.stdout


def test_changelog_for_series(set_env_client):
    with _mock_urlopen():
        res = run_cli_command(["changelog", "--for", "1.2"])
    assert "### 1.2.5" in res.stdout
    assert "### 1.2.4" in res.stdout
    assert "### 1.2.0" in res.stdout
    assert "### 1.3" not in res.stdout


def test_changelog_for_exact(set_env_client):
    with _mock_urlopen():
        res = run_cli_command(["changelog", "--for", "1.2.5"])
    assert "### 1.2.5" in res.stdout
    assert "### 1.2.4" not in res.stdout
    assert "### 1.3" not in res.stdout


def test_changelog_all(set_env_client):
    with _mock_urlopen():
        res = run_cli_command(["changelog", "--all"])
    assert "### 1.3.3" in res.stdout
    assert "### 1.2.0" in res.stdout


def test_changelog_json(set_env_client):
    with _mock_urlopen():
        res = run_cli_command(["changelog", "--all", "--json"])
    data = json.loads(res.stdout)
    assert isinstance(data, list)
    assert len(data) == 6
    assert data[0]["version"] == "1.3.3"
    assert data[0]["date"] == "2026-02-12"
    assert "volume mounts" in data[0]["body"]


def test_changelog_newer(set_env_client):
    with _mock_urlopen(), mock.patch("modal_version.__version__", "1.3.0"):
        res = run_cli_command(["changelog", "--newer"])
    assert "### 1.3.3" in res.stdout
    assert "### 1.3.2" in res.stdout
    assert "### 1.3.0" not in res.stdout
    assert "### 1.2" not in res.stdout


def test_changelog_no_entries(set_env_client):
    with _mock_urlopen(), mock.patch("modal_version.__version__", "9.9.9"):
        res = run_cli_command(["changelog"])
    assert "No changelog entries found" in res.stdout
