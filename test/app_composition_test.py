# Copyright Modal Labs 2024
from test.helpers import deploy_app_externally


def test_app_composition_includes_all_functions(servicer, supports_dir, monkeypatch, client):
    print(deploy_app_externally(servicer, "main.py", cwd=supports_dir / "multifile_project"))
    assert servicer.n_functions == 3
    assert {"/root/main.py", "/root/a.py", "/root/b.py", "/root/c.py"} == set(servicer.files_name2sha.keys())
    assert len(servicer.secrets) == 1  # secret from B should be included
    assert servicer.n_mounts == 4  # mounts should not be duplicated
