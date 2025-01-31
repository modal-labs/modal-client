# Copyright Modal Labs 2025
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from modal._utils.docker_utils import extract_copy_command_patterns, find_dockerignore_file


@pytest.mark.parametrize(
    ("copy_commands", "expected_patterns"),
    [
        (
            ["CoPY files/dir1 ./smth_copy"],
            {"files/dir1"},
        ),
        (
            ["COPY files/*.txt /dest/", "COPY files/**/*.py /dest/"],
            {"files/*.txt", "files/**/*.py"},
        ),
        (
            ["COPY files/special/file[[]1].txt /dest/"],
            {"files/special/file[[]1].txt"},
        ),
        (
            ["COPY files/*.txt files/**/*.py /dest/"],
            {"files/*.txt", "files/**/*.py"},
        ),
        (
            [
                "copy --from=a b",
                "copy ./smth \\",
                "./foo.py \\",
                "# this is a comment",
                "./bar.py \\",
                "/x",
            ],
            {"./smth", "./foo.py", "./bar.py"},
        ),
        (
            [
                "COPY --from=a b",
            ],
            set(),
        ),
    ],
)
def test_extract_copy_command_patterns(copy_commands, expected_patterns):
    copy_command_sources = set(extract_copy_command_patterns(copy_commands))
    assert copy_command_sources == expected_patterns


@pytest.mark.usefixtures("tmp_cwd")
def test_image_dockerfile_copy_messy():
    with TemporaryDirectory(dir="./") as tmp_dir:
        dockerfile = NamedTemporaryFile("w", delete=False)
        dockerfile.write(
            f"""
FROM python:3.12-slim

WORKDIR /my-app

RUN ls

# COPY simple directory
    CoPY {tmp_dir}/dir1 ./smth_copy

RUN ls -la

# COPY multiple sources
        COPY {tmp_dir}/test.py {tmp_dir}/file10.txt /

RUN ls \\
    -l

# COPY multiple lines
copy {tmp_dir}/dir2 \\
    {tmp_dir}/file1.txt \\
# this is a comment
    {tmp_dir}/file2.txt \\
    /x

        RUN ls
        """
        )
        dockerfile.close()

        with open(dockerfile.name) as f:
            lines = f.readlines()

        assert sorted(extract_copy_command_patterns(lines)) == sorted(
            [
                f"{tmp_dir}/dir1",
                f"{tmp_dir}/test.py",
                f"{tmp_dir}/file10.txt",
                f"{tmp_dir}/dir2",
                f"{tmp_dir}/file1.txt",
                f"{tmp_dir}/file2.txt",
            ]
        )


@pytest.mark.usefixtures("tmp_cwd")
def test_find_generic_cwd_dockerignore_file():
    test_cwd = Path.cwd()
    with TemporaryDirectory(dir=test_cwd) as tmp_dir:
        tmp_path = Path(tmp_dir)
        dir1 = tmp_path / "dir1"
        dir1.mkdir()

        dockerfile_path = dir1 / "Dockerfile"
        dockerignore_path = tmp_path / ".dockerignore"
        dockerignore_path.write_text("**/*")
        assert find_dockerignore_file(test_cwd / tmp_dir, dockerfile_path) == dockerignore_path


@pytest.mark.usefixtures("tmp_cwd")
def test_dont_find_specific_dockerignore_file():
    test_cwd = Path.cwd()
    with TemporaryDirectory(dir=test_cwd) as tmp_dir:
        tmp_path = Path(tmp_dir)
        dir1 = tmp_path / "dir1"
        dir1.mkdir()

        dockerfile_path = dir1 / "foo"
        dockerignore_path = tmp_path / "foo.dockerignore"
        dockerignore_path.write_text("**/*")
        assert find_dockerignore_file(test_cwd / tmp_dir, dockerfile_path) is None


@pytest.mark.usefixtures("tmp_cwd")
def test_prefer_specific_cwd_dockerignore_file():
    test_cwd = Path.cwd()
    with TemporaryDirectory(dir=test_cwd) as tmp_dir:
        tmp_path = Path(tmp_dir)
        dir1 = tmp_path / "dir1"
        dir1.mkdir()

        dockerfile_path = tmp_path / "Dockerfile"
        generic_dockerignore_path = tmp_path / ".dockerignore"
        generic_dockerignore_path.write_text("**/*.py")
        specific_dockerignore_path = tmp_path / "Dockerfile.dockerignore"
        specific_dockerignore_path.write_text("**/*")
        assert find_dockerignore_file(test_cwd / tmp_dir, dockerfile_path) == specific_dockerignore_path
        assert find_dockerignore_file(test_cwd / tmp_dir, dockerfile_path) != generic_dockerignore_path


@pytest.mark.usefixtures("tmp_cwd")
def test_dont_find_nested_dockerignore_file():
    test_cwd = Path.cwd()
    with TemporaryDirectory(dir=test_cwd) as tmp_dir:
        tmp_path = Path(tmp_dir)
        dir1 = tmp_path / "dir1"
        dir1.mkdir()
        dir2 = dir1 / "dir2"
        dir2.mkdir()

        dockerfile_path = dir1 / "Dockerfile"
        dockerfile_path.write_text("COPY . /dummy")

        # should ignore parent ones
        generic_dockerignore_path = tmp_path / ".dockerignore"
        generic_dockerignore_path.write_text("**/*")
        specific_dockerignore_path = tmp_path / "Dockerfile.dockerignore"
        specific_dockerignore_path.write_text("**/*")

        # should ignore nested ones
        nested_generic_dockerignore_path = dir2 / ".dockerignore"
        nested_generic_dockerignore_path.write_text("**/*")
        nested_specific_dockerignore_path = dir2 / "Dockerfile.dockerignore"
        nested_specific_dockerignore_path.write_text("**/*")

        assert find_dockerignore_file(dir1, dockerfile_path) is None


@pytest.mark.usefixtures("tmp_cwd")
def test_find_next_to_dockerfile_dockerignore_file():
    test_cwd = Path.cwd()
    with TemporaryDirectory(dir=test_cwd) as tmp_dir:
        tmp_path = Path(tmp_dir)
        dir1 = tmp_path / "dir1"
        dir1.mkdir()

        dockerfile_path = dir1 / "Dockerfile"
        dockerignore_path = tmp_path / ".dockerignore"
        dockerignore_path.write_text("**/*")

        assert find_dockerignore_file(test_cwd / tmp_dir, dockerfile_path) == dockerignore_path
