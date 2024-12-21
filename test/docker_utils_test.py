# Copyright Modal Labs 2024
import pytest
from tempfile import NamedTemporaryFile, TemporaryDirectory

from modal._utils.docker_utils import extract_copy_command_patterns


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
                "copy ./smth \\",
                "./foo.py \\",
                "# this is a comment",
                "./bar.py \\",
                "/x",
            ],
            {"./smth", "./foo.py", "./bar.py"},
        ),
    ],
)
def test_extract_copy_command_patterns(copy_commands, expected_patterns):
    copy_command_sources = set(extract_copy_command_patterns(copy_commands))
    assert copy_command_sources == expected_patterns


@pytest.mark.usefixture("tmp_cwd")
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
