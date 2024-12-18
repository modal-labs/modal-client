# Copyright Modal Labs 2024
"""Pattern matching library tests ported from https://github.com/golang/go/blob/go1.23.4/src/path/match_test.go"""

import platform
import pytest

from modal._utils.match import PatternError, match


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
        ("[]a]", "]", False, PatternError),
        ("[-]", "-", False, PatternError),
        ("[x-]", "x", False, PatternError),
        ("[x-]", "-", False, PatternError),
        ("[x-]", "z", False, PatternError),
        ("[-x]", "x", False, PatternError),
        ("[-x]", "-", False, PatternError),
        ("[-x]", "a", False, PatternError),
        ("\\", "a", False, PatternError),
        ("[a-b-c]", "a", False, PatternError),
        ("[", "a", False, PatternError),
        ("[^", "a", False, PatternError),
        ("[^bc", "a", False, PatternError),
        ("a[", "a", False, PatternError),
        ("a[", "ab", False, PatternError),
        ("a[", "x", False, PatternError),
        ("a/b[", "x", False, PatternError),
        ("*x", "xxx", True, None),
    ]


@pytest.mark.parametrize("pattern, s, is_match, err", match_tests())
def test_match(pattern, s, is_match, err):
    if platform.system() == "Windows" and "\\" in pattern:
        # No escape allowed on Windows.
        return

    if err is not None:
        with pytest.raises(ValueError):
            match(pattern, s)
    else:
        actual_match = match(pattern, s)
        assert is_match == actual_match, f"{pattern=} {s=} {is_match=} {actual_match=}"
