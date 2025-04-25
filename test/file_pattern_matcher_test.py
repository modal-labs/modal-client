# Copyright Modal Labs 2024

"""Tests for file_pattern_matcher.py.

These are ported from the original patternmatcher Go library.
"""

import contextlib
import os
import os.path
import platform
import pytest
from pathlib import Path

from modal import FilePatternMatcher


def test_wildcard_matches():
    assert FilePatternMatcher("*")._matches("fileutils.go")


def test_pattern_matches():
    assert FilePatternMatcher("*.go")._matches("fileutils.go")


def test_exclusion_pattern_matches_pattern_before():
    assert FilePatternMatcher("!fileutils.go", "*.go")._matches("fileutils.go")


def test_pattern_matches_folder_exclusions():
    assert not FilePatternMatcher("docs", "!docs/README.md")._matches("docs/README.md")


def test_pattern_matches_folder_with_slash_exclusions():
    assert not FilePatternMatcher("docs/", "!docs/README.md")._matches("docs/README.md")


def test_pattern_matches_folder_wildcard_exclusions():
    assert not FilePatternMatcher("docs/*", "!docs/README.md")._matches("docs/README.md")


def test_exclusion_pattern_matches_pattern_after():
    assert not FilePatternMatcher("*.go", "!fileutils.go")._matches("fileutils.go")


def test_exclusion_pattern_matches_whole_directory():
    assert not FilePatternMatcher("*.go")._matches(".")


def test_single_exclamation_error():
    try:
        FilePatternMatcher("!")
    except ValueError as e:
        assert str(e) == 'Illegal exclusion pattern: "!"'


def test_matches_with_no_patterns():
    assert not FilePatternMatcher()._matches("/any/path/there")


def test_matches_with_malformed_patterns():
    try:
        FilePatternMatcher("[")
    except ValueError as e:
        assert str(e) == "Bad pattern: ["


def test_matches():
    tests = [
        ("**", "file", True),
        ("**", "file/", True),
        ("**/", "file", True),  # weird one
        ("**/", "file/", True),
        ("**", "/", True),
        ("**/", "/", True),
        ("**", "dir/file", True),
        ("**/", "dir/file", True),
        ("**", "dir/file/", True),
        ("**/", "dir/file/", True),
        ("**/**", "dir/file", True),
        ("**/**", "dir/file/", True),
        ("dir/**", "dir/file", True),
        ("dir/**", "dir/file/", True),
        ("dir/**", "dir/dir2/file", True),
        ("dir/**", "dir/dir2/file/", True),
        ("**/dir", "dir", True),
        ("**/dir", "dir/file", True),
        ("**/dir2/*", "dir/dir2/file", True),
        ("**/dir2/*", "dir/dir2/file/", True),
        ("**/dir2/**", "dir/dir2/dir3/file", True),
        ("**/dir2/**", "dir/dir2/dir3/file/", True),
        ("**file", "file", True),
        ("**file", "dir/file", True),
        ("**/file", "dir/file", True),
        ("**file", "dir/dir/file", True),
        ("**/file", "dir/dir/file", True),
        ("**/file*", "dir/dir/file", True),
        ("**/file*", "dir/dir/file.txt", True),
        ("**/file*txt", "dir/dir/file.txt", True),
        ("**/file*.txt", "dir/dir/file.txt", True),
        ("**/file*.txt*", "dir/dir/file.txt", True),
        ("**/**/*.txt", "dir/dir/file.txt", True),
        ("**/**/*.txt2", "dir/dir/file.txt", False),
        ("**/*.txt", "file.txt", True),
        ("**/**/*.txt", "file.txt", True),
        ("a**/*.txt", "a/file.txt", True),
        ("a**/*.txt", "a/dir/file.txt", True),
        ("a**/*.txt", "a/dir/dir/file.txt", True),
        ("a/*.txt", "a/dir/file.txt", False),
        ("a/*.txt", "a/file.txt", True),
        ("a/*.txt**", "a/file.txt", True),
        ("a[b-d]e", "ae", False),
        ("a[b-d]e", "ace", True),
        ("a[b-d]e", "aae", False),
        ("a[^b-d]e", "aze", True),
        (".*", ".foo", True),
        (".*", "foo", False),
        ("abc.def", "abcdef", False),
        ("abc.def", "abc.def", True),
        ("abc.def", "abcZdef", False),
        ("abc?def", "abcZdef", True),
        ("abc?def", "abcdef", False),
        ("a\\\\", "a\\", True),
        ("**/foo/bar", "foo/bar", True),
        ("**/foo/bar", "dir/foo/bar", True),
        ("**/foo/bar", "dir/dir2/foo/bar", True),
        ("abc/**", "abc", False),
        ("abc/**", "abc/def", True),
        ("abc/**", "abc/def/ghi", True),
        ("**/.foo", ".foo", True),
        ("**/.foo", "bar.foo", False),
        ("a(b)c/def", "a(b)c/def", True),
        ("a(b)c/def", "a(b)c/xyz", False),
        ("a.|)$(}+{bc", "a.|)$(}+{bc", True),
        (
            "dist/proxy.py-2.4.0rc3.dev36+g08acad9-py3-none-any.whl",
            "dist/proxy.py-2.4.0rc3.dev36+g08acad9-py3-none-any.whl",
            True,
        ),
        ("dist/*.whl", "dist/proxy.py-2.4.0rc3.dev36+g08acad9-py3-none-any.whl", True),
    ]

    multi_pattern_tests = [
        (["**", "!util/docker/web"], "util/docker/web/foo", False),
        (["**", "!util/docker/web", "util/docker/web/foo"], "util/docker/web/foo", True),
        (
            ["**", "!dist/proxy.py-2.4.0rc3.dev36+g08acad9-py3-none-any.whl"],
            "dist/proxy.py-2.4.0rc3.dev36+g08acad9-py3-none-any.whl",
            False,
        ),
        (["**", "!dist/*.whl"], "dist/proxy.py-2.4.0rc3.dev36+g08acad9-py3-none-any.whl", False),
    ]

    for pattern, text, expected in tests:
        assert FilePatternMatcher(pattern)._matches(text) is expected

    for patterns, text, expected in multi_pattern_tests:
        assert FilePatternMatcher(*patterns)._matches(text) is expected


def test_clean_patterns():
    patterns = ["docs", "config"]
    pm = FilePatternMatcher(*patterns)
    cleaned = pm.patterns
    assert len(cleaned) == 2


def test_clean_patterns_strip_empty_patterns():
    patterns = ["docs", "config", ""]
    pm = FilePatternMatcher(*patterns)
    cleaned = pm.patterns
    assert len(cleaned) == 2


def test_clean_patterns_exception_flag():
    patterns = ["docs", "!docs/README.md"]
    pm = FilePatternMatcher(*patterns)
    assert any(p.exclusion for p in pm.patterns)


def test_clean_patterns_leading_space_trimmed():
    patterns = ["docs", "  !docs/README.md"]
    pm = FilePatternMatcher(*patterns)
    assert any(p.exclusion for p in pm.patterns)


def test_clean_patterns_trailing_space_trimmed():
    patterns = ["docs", "!docs/README.md  "]
    pm = FilePatternMatcher(*patterns)
    assert any(p.exclusion for p in pm.patterns)


def test_clean_patterns_error_single_exception():
    patterns = ["!"]
    try:
        FilePatternMatcher(*patterns)
    except ValueError as e:
        assert str(e) == 'Illegal exclusion pattern: "!"'


def test_match():
    match_tests = [
        ("abc", "abc", True, None),
        ("*", "abc", True, None),
        ("*c", "abc", True, None),
        ("a*", "a", True, None),
        ("a*", "abc", True, None),
        ("a*", "ab/c", True, None),
        ("a*/b", "abc/b", True, None),
        ("a*/b", "a/c/b", False, None),
        ("a*b*c*d*e*/f", "axbxcxdxe/f", True, None),
        ("a*b*c*d*e*/f", "axbxcxdxexxx/f", True, None),
        ("a*b*c*d*e*/f", "axbxcxdxe/xxx/f", False, None),
        ("a*b*c*d*e*/f", "axbxcxdxexxx/fff", False, None),
        ("a*b?c*x", "abxbbxdbxebxczzx", True, None),
        ("a*b?c*x", "abxbbxdbxebxczzy", False, None),
        ("ab[c]", "abc", True, None),
        ("ab[b-d]", "abc", True, None),
        ("ab[e-g]", "abc", False, None),
        ("ab[^c]", "abc", False, None),
        ("ab[^b-d]", "abc", False, None),
        ("ab[^e-g]", "abc", True, None),
        ("a\\*b", "a*b", True, None),
        ("a\\*b", "ab", False, None),
        ("a?b", "a☺b", True, None),
        ("a[^a]b", "a☺b", True, None),
        ("a???b", "a☺b", False, None),
        ("a[^a][^a][^a]b", "a☺b", False, None),
        ("[a-ζ]*", "α", True, None),
        ("*[a-ζ]", "A", False, None),
        ("a?b", "a/b", False, None),
        ("a*b", "a/b", False, None),
        ("[\\]a]", "]", True, None),
        ("[\\-]", "-", True, None),
        ("[x\\-]", "x", True, None),
        ("[x\\-]", "-", True, None),
        ("[x\\-]", "z", False, None),
        ("[\\-x]", "x", True, None),
        ("[\\-x]", "-", True, None),
        ("[\\-x]", "a", False, None),
        # These do not return errors because the Python re.compile() method does
        # not raise an error on invalid syntax like Go does. We can omit the
        # tests though since it doesn't affect behavior on _correct_ syntax.
        #
        # ("[]a]", "]", False, ValueError),
        # ("[-]", "-", False, ValueError),
        # ("[x-]", "x", False, ValueError),
        # ("[x-]", "-", False, ValueError),
        # ("[x-]", "z", False, ValueError),
        # ("[-x]", "x", False, ValueError),
        # ("[-x]", "-", False, ValueError),
        # ("[-x]", "a", False, ValueError),
        # ("\\", "a", False, ValueError),
        # ("[a-b-c]", "a", False, ValueError),
        # ("[", "a", False, ValueError),
        # ("[^", "a", False, ValueError),
        # ("[^bc", "a", False, ValueError),
        # ("a[", "a", False, ValueError),
        # ("a[", "ab", False, ValueError),
        ("*x", "xxx", True, None),
    ]

    for pattern, text, expected, error in match_tests:
        if platform.system() == "Windows":
            if "\\" in pattern:
                # No escape allowed on Windows.
                continue
            pattern = os.path.normpath(pattern)
            text = os.path.normpath(text)

        with pytest.raises(error) if error else contextlib.nullcontext():
            assert FilePatternMatcher(pattern)._matches(text) is expected


def __helper_get_file_paths(tmp_path: Path) -> list[Path]:
    file_paths = []
    for root, _, files in os.walk(tmp_path):
        for file in files:
            file_paths.append(Path(os.path.join(root, file)))
    return file_paths


def test_against_paths(tmp_path_with_content):
    tmp_path = tmp_path_with_content
    file_paths = __helper_get_file_paths(tmp_path)

    # match everything that's not ignored
    lff = FilePatternMatcher("**/*", "!**/module")

    for file_path in file_paths:
        if "module" in str(file_path):
            assert not lff(file_path)
        else:
            assert lff(file_path)

    lff = FilePatternMatcher("**/*.py")

    for file_path in file_paths:
        if str(file_path).endswith(".py"):
            assert lff(file_path)
        else:
            assert not lff(file_path)


def test_empty_patterns(tmp_path_with_content):
    tmp_path = tmp_path_with_content
    file_paths = __helper_get_file_paths(tmp_path)
    lff = FilePatternMatcher()

    for file_path in file_paths:
        assert not lff(file_path)


def test_invert_patterns(tmp_path_with_content):
    tmp_path = tmp_path_with_content
    file_paths = __helper_get_file_paths(tmp_path)

    # match everything that's not ignored
    lff = ~FilePatternMatcher("**/*", "!**/module")

    for file_path in file_paths:
        if "module" in str(file_path):
            assert lff(file_path)
        else:
            assert not lff(file_path)

    # empty patterns should match nothing
    # inverted empty patterns should match everything
    lff = ~FilePatternMatcher()
    for file_path in file_paths:
        assert lff(file_path)

    # single negative pattern should match nothing
    lff = FilePatternMatcher("!**/*.txt")
    for file_path in file_paths:
        assert not lff(file_path)


@pytest.mark.usefixtures("tmp_cwd")
@pytest.mark.parametrize("as_type", [Path, str])
def test_from_file(as_type):
    rel_top_dir = Path("top")
    rel_top_dir.mkdir()
    ignore_file = rel_top_dir / "pattern_file"
    ignore_file.write_text("**/*.txt")

    lff = FilePatternMatcher.from_file(as_type(ignore_file))
    assert lff(Path("top/data.txt"))
    assert not lff(Path("top/data.py"))
