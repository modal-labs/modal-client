from test.helpers import deploy_stub_externally


def test_stub_composition_includes_all_functions(servicer, supports_dir, monkeypatch, client):
    deploy_stub_externally(servicer, "main.py", cwd=supports_dir / "multifile_project")
    assert servicer.n_functions == 2
    assert {"/root/main.py", "/root/a.py", "/root/b.py"} == set(servicer.files_name2sha.keys())
    assert len(servicer.secrets) == 1  # secret from B should be included
    #  assert servicer.n_mounts == 3 TODO: uncomment after stub deduplication improvements have been merged
