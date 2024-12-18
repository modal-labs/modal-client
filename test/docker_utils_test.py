# Copyright Modal Labs 2024
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from modal._utils.docker_utils import extract_copy_command_patterns, find_dockerignore_file


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


def test_find_dockerignore_file():
    print()
    test_cwd = Path.cwd()

    # case 1:
    # generic dockerignore file in cwd --> find it
    with TemporaryDirectory(dir=test_cwd) as tmp_dir:
        tmp_path = Path(tmp_dir)
        dir1 = tmp_path / "dir1"
        dir1.mkdir()

        os.chdir(test_cwd / tmp_dir)

        dockerfile_path = dir1 / "Dockerfile"
        dockerignore_path = tmp_path / ".dockerignore"
        dockerignore_path.write_text("**/*")
        assert find_dockerignore_file(dockerfile_path) == dockerignore_path

    # case 2:
    # specific dockerignore file in cwd --> find it
    with TemporaryDirectory(dir=test_cwd) as tmp_dir:
        tmp_path = Path(tmp_dir)
        dir1 = tmp_path / "dir1"
        dir1.mkdir()

        os.chdir(test_cwd / tmp_dir)

        dockerfile_path = dir1 / "foo"
        dockerignore_path = tmp_path / "foo.dockerignore"
        dockerignore_path.write_text("**/*")
        assert find_dockerignore_file(dockerfile_path) == dockerignore_path

    # case 3:
    # generic dockerignore file and nested dockerignore file in cwd
    # should match specific
    with TemporaryDirectory(dir=test_cwd) as tmp_dir:
        tmp_path = Path(tmp_dir)
        dir1 = tmp_path / "dir1"
        dir1.mkdir()

        os.chdir(test_cwd / tmp_dir)

        dockerfile_path = tmp_path / "Dockerfile"
        generic_dockerignore_path = tmp_path / ".dockerignore"
        generic_dockerignore_path.write_text("**/*.py")
        specific_dockerignore_path = tmp_path / "Dockerfile.dockerignore"
        specific_dockerignore_path.write_text("**/*")
        assert find_dockerignore_file(dockerfile_path) == specific_dockerignore_path
        assert find_dockerignore_file(dockerfile_path) != generic_dockerignore_path

    # case 4:
    # should not match nested dockerignore files
    # or parent dockerignore files
    # when dockerfile is in cwd
    with TemporaryDirectory(dir=test_cwd) as tmp_dir:
        tmp_path = Path(tmp_dir)
        dir1 = tmp_path / "dir1"
        dir1.mkdir()
        dir2 = dir1 / "dir2"
        dir2.mkdir()

        os.chdir(dir1)

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

        assert find_dockerignore_file(dockerfile_path) is None

    # case 5:
    # should match dockerignore file next to dockerfile
    # and not next to cwd if both exist
    # even if more specific
    with TemporaryDirectory(dir=test_cwd) as tmp_dir:
        tmp_path = Path(tmp_dir)
        dir1 = tmp_path / "dir1"
        dir1.mkdir()

        os.chdir(test_cwd / tmp_dir)

        dockerfile_path = dir1 / "Dockerfile"
        dockerignore_path = tmp_path / ".dockerignore"
        dockerignore_path.write_text("**/*")
        assert find_dockerignore_file(dockerfile_path) == dockerignore_path
