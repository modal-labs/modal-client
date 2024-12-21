# Copyright Modal Labs 2024
import os
import pytest
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from modal._utils.docker_utils import extract_copy_command_patterns


def test_extract_copy_command_patterns():
    x = [
        (
            ["CoPY files/dir1 ./smth_copy"],
            ["files/dir1"],
        ),
        (
            ["COPY files/*.txt /dest/", "COPY files/**/*.py /dest/"],
            ["files/*.txt", "files/**/*.py"],
        ),
        (
            ["COPY files/special/file[[]1].txt /dest/"],
            ["files/special/file[[]1].txt"],
        ),
        (
            ["COPY files/*.txt files/**/*.py /dest/"],
            ["files/*.txt", "files/**/*.py"],
        ),
        (
            [
                "copy ./smth \\",
                "./foo.py \\",
                "# this is a comment",
                "./bar.py \\",
                "/x",
            ],
            ["./smth", "./foo.py", "./bar.py"],
        ),
    ]

    for dockerfile_lines, expected in x:
        copy_command_sources = sorted(extract_copy_command_patterns(dockerfile_lines))
        expected = sorted(expected)
        assert copy_command_sources == expected


def create_tmp_files(tmp_path):
    (tmp_path / "dir1").mkdir()
    (tmp_path / "dir1" / "a.txt").write_text("a")
    (tmp_path / "dir1" / "b.txt").write_text("b")

    (tmp_path / "dir2").mkdir()
    (tmp_path / "dir2" / "test1.py").write_text("test1")
    (tmp_path / "dir2" / "test2.py").write_text("test2")

    (tmp_path / "file1.txt").write_text("content1")
    (tmp_path / "file10.txt").write_text("content1")
    (tmp_path / "file2.txt").write_text("content2")
    (tmp_path / "test.py").write_text("python")

    (tmp_path / "special").mkdir()
    (tmp_path / "special" / "file[1].txt").write_text("special1")
    (tmp_path / "special" / "file{2}.txt").write_text("special2")

    if sys.platform != "win32":
        (tmp_path / "special" / "test?file.py").write_text("special3")

    (tmp_path / "this").mkdir()
    (tmp_path / "this" / "is").mkdir()
    (tmp_path / "this" / "is" / "super").mkdir()
    (tmp_path / "this" / "is" / "super" / "nested").mkdir()
    (tmp_path / "this" / "is" / "super" / "nested" / "file.py").write_text("python")

    all_fps = []
    for root, _, files in os.walk(tmp_path):
        for file in files:
            all_fps.append(f"{os.path.join(root, file)}".lstrip("./"))

    return all_fps


@pytest.mark.usefixture("tmp_cwd")
def test_image_dockerfile_copy_messy():
    with TemporaryDirectory(dir="./") as tmp_dir:
        tmp_path = Path(tmp_dir)

        create_tmp_files(tmp_path)

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
