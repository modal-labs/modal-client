# Copyright Modal Labs 2024
"""Pattern matching library tests ported from https://github.com/golang/go/blob/go1.23.4/src/path/match_test.go"""

import platform
import pytest

from modal._utils.docker_copy_match import dockerfile_match


def match_tests():
    return [
        # (pattern, string, is_match, error)
        ("abc", "abc", True, None),
        ("*", "abc", True, None),
        ("*c", "abc", True, None),
        ("a*", "a", True, None),
        ("a*", "abc", True, None),
        ("a*", "ab/c", False, None),
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
        ("[]a]", "]", False, ValueError),
        ("[-]", "-", False, ValueError),
        ("[x-]", "x", False, ValueError),
        ("[x-]", "-", False, ValueError),
        ("[x-]", "z", False, ValueError),
        ("[-x]", "x", False, ValueError),
        ("[-x]", "-", False, ValueError),
        ("[-x]", "a", False, ValueError),
        ("\\", "a", False, ValueError),
        ("[a-b-c]", "a", False, ValueError),
        ("[", "a", False, ValueError),
        ("[^", "a", False, ValueError),
        ("[^bc", "a", False, ValueError),
        ("a[", "a", False, ValueError),
        ("a[", "ab", False, ValueError),
        ("a[", "x", False, ValueError),
        ("a/b[", "x", False, ValueError),
        ("*x", "xxx", True, None),
    ]


def test_match():
    for pattern, s, expected_match, err in match_tests():
        if platform.system() == "Windows" and "\\" in pattern:
            # No escape allowed on Windows.
            return

        if err is not None:
            with pytest.raises(ValueError):
                dockerfile_match(pattern, s)
        else:
            actual_match = dockerfile_match(pattern, s)
            assert expected_match == actual_match, f"{pattern=} {s=} {expected_match=} {actual_match=}"
