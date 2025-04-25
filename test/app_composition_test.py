# Copyright Modal Labs 2024
from test.helpers import deploy_app_externally


def test_app_composition_includes_all_functions(servicer, credentials, supports_dir, monkeypatch, client):
    print(deploy_app_externally(servicer, credentials, "multifile_project.main", cwd=supports_dir))
    assert servicer.n_functions == 5
    assert {
        "/root/multifile_project/__init__.py",
        "/root/multifile_project/main.py",
        "/root/multifile_project/a.py",
        "/root/multifile_project/b.py",
        "/root/multifile_project/c.py",
    } == set(servicer.files_name2sha.keys())
    assert len(servicer.secrets) == 1  # secret from B should be included
    assert servicer.n_mounts == 1  # mounts should not be duplicated, and the automount for the package includes all
