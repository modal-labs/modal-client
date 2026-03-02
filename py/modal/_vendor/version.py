"""
Vendored version of packaging.version v26.0.

We vendor only version.py, plus a couple of types from _structures.py
which we moved into this file so that we vendor a single module.

---

Dual-licensed under Apache v2 and BSD-2.

Copyright (c) Donald Stufft and individual contributors.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

"""
from __future__ import annotations

import itertools
import re
from typing import Any, Callable, NamedTuple, SupportsInt, Tuple, Union


__all__ = ["VERSION_PATTERN", "InvalidVersion", "Version", "parse"]

class InfinityType:
    def __repr__(self) -> str:
        return "Infinity"

    def __hash__(self) -> int:
        return hash(repr(self))

    def __lt__(self, other: object) -> bool:
        return False

    def __le__(self, other: object) -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __gt__(self, other: object) -> bool:
        return True

    def __ge__(self, other: object) -> bool:
        return True

    def __neg__(self: object) -> "NegativeInfinityType":
        return NegativeInfinity


Infinity = InfinityType()


class NegativeInfinityType:
    def __repr__(self) -> str:
        return "-Infinity"

    def __hash__(self) -> int:
        return hash(repr(self))

    def __lt__(self, other: object) -> bool:
        return True

    def __le__(self, other: object) -> bool:
        return True

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __gt__(self, other: object) -> bool:
        return False

    def __ge__(self, other: object) -> bool:
        return False

    def __neg__(self: object) -> InfinityType:
        return Infinity


NegativeInfinity = NegativeInfinityType()

LocalType = Tuple[Union[int, str], ...]

CmpPrePostDevType = Union[InfinityType, NegativeInfinityType, Tuple[str, int]]
CmpLocalType = Union[
    NegativeInfinityType,
    Tuple[Union[Tuple[int, str], Tuple[NegativeInfinityType, Union[int, str]]], ...],
]
CmpKey = Tuple[
    int,
    Tuple[int, ...],
    CmpPrePostDevType,
    CmpPrePostDevType,
    CmpPrePostDevType,
    CmpLocalType,
]
VersionComparisonMethod = Callable[[CmpKey, CmpKey], bool]


class _Version(NamedTuple):
    epoch: int
    release: tuple[int, ...]
    dev: tuple[str, int] | None
    pre: tuple[str, int] | None
    post: tuple[str, int] | None
    local: LocalType | None


def parse(version: str) -> Version:
    """Parse the given version string.

    >>> parse('1.0.dev1')
    <Version('1.0.dev1')>

    :param version: The version string to parse.
    :raises InvalidVersion: When the version string is not a valid version.
    """
    return Version(version)


class InvalidVersion(ValueError):
    """Raised when a version string is not a valid version.

    >>> Version("invalid")
    Traceback (most recent call last):
        ...
    packaging.version.InvalidVersion: Invalid version: 'invalid'
    """


class _BaseVersion:
    _key: tuple[Any, ...]

    def __hash__(self) -> int:
        return hash(self._key)

    # Please keep the duplicated `isinstance` check
    # in the six comparisons hereunder
    # unless you find a way to avoid adding overhead function calls.
    def __lt__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key < other._key

    def __le__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key <= other._key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key == other._key

    def __ge__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key >= other._key

    def __gt__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key > other._key

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key != other._key


# Deliberately not anchored to the start and end of the string, to make it
# easier for 3rd party code to reuse
_VERSION_PATTERN = r"""
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?
            (?P<pre_l>alpha|a|beta|b|preview|pre|c|rc)
            [-_\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)
                [-_\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          # dev release
            [-_\.]?
            (?P<dev_l>dev)
            [-_\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
"""

VERSION_PATTERN = _VERSION_PATTERN
"""
A string containing the regular expression used to match a valid version.

The pattern is not anchored at either end, and is intended for embedding in larger
expressions (for example, matching a version number as part of a file name). The
regular expression should be compiled with the ``re.VERBOSE`` and ``re.IGNORECASE``
flags set.

:meta hide-value:
"""


class Version(_BaseVersion):
    """This class abstracts handling of a project's versions.

    A :class:`Version` instance is comparison aware and can be compared and
    sorted using the standard Python interfaces.

    >>> v1 = Version("1.0a5")
    >>> v2 = Version("1.0")
    >>> v1
    <Version('1.0a5')>
    >>> v2
    <Version('1.0')>
    >>> v1 < v2
    True
    >>> v1 == v2
    False
    >>> v1 > v2
    False
    >>> v1 >= v2
    False
    >>> v1 <= v2
    True
    """

    _regex = re.compile(r"^\s*" + VERSION_PATTERN + r"\s*$", re.VERBOSE | re.IGNORECASE)
    _key: CmpKey

    def __init__(self, version: str) -> None:
        """Initialize a Version object.

        :param version:
            The string representation of a version which will be parsed and normalized
            before use.
        :raises InvalidVersion:
            If the ``version`` does not conform to PEP 440 in any way then this
            exception will be raised.
        """

        # Validate the version and parse it into pieces
        match = self._regex.search(version)
        if not match:
            raise InvalidVersion(f"Invalid version: {version!r}")

        # Store the parsed out pieces of the version
        self._version = _Version(
            epoch=int(match.group("epoch")) if match.group("epoch") else 0,
            release=tuple(int(i) for i in match.group("release").split(".")),
            pre=_parse_letter_version(match.group("pre_l"), match.group("pre_n")),
            post=_parse_letter_version(
                match.group("post_l"), match.group("post_n1") or match.group("post_n2")
            ),
            dev=_parse_letter_version(match.group("dev_l"), match.group("dev_n")),
            local=_parse_local_version(match.group("local")),
        )

        # Generate a key which will be used for sorting
        self._key = _cmpkey(
            self._version.epoch,
            self._version.release,
            self._version.pre,
            self._version.post,
            self._version.dev,
            self._version.local,
        )

    def __repr__(self) -> str:
        """A representation of the Version that shows all internal state.

        >>> Version('1.0.0')
        <Version('1.0.0')>
        """
        return f"<Version('{self}')>"

    def __str__(self) -> str:
        """A string representation of the version that can be round-tripped.

        >>> str(Version("1.0a5"))
        '1.0a5'
        """
        parts = []

        # Epoch
        if self.epoch != 0:
            parts.append(f"{self.epoch}!")

        # Release segment
        parts.append(".".join(str(x) for x in self.release))

        # Pre-release
        if self.pre is not None:
            parts.append("".join(str(x) for x in self.pre))

        # Post-release
        if self.post is not None:
            parts.append(f".post{self.post}")

        # Development release
        if self.dev is not None:
            parts.append(f".dev{self.dev}")

        # Local version segment
        if self.local is not None:
            parts.append(f"+{self.local}")

        return "".join(parts)

    @property
    def epoch(self) -> int:
        """The epoch of the version.

        >>> Version("2.0.0").epoch
        0
        >>> Version("1!2.0.0").epoch
        1
        """
        return self._version.epoch

    @property
    def release(self) -> tuple[int, ...]:
        """The components of the "release" segment of the version.

        >>> Version("1.2.3").release
        (1, 2, 3)
        >>> Version("2.0.0").release
        (2, 0, 0)
        >>> Version("1!2.0.0.post0").release
        (2, 0, 0)

        Includes trailing zeroes but not the epoch or any pre-release / development /
        post-release suffixes.
        """
        return self._version.release

    @property
    def pre(self) -> tuple[str, int] | None:
        """The pre-release segment of the version.

        >>> print(Version("1.2.3").pre)
        None
        >>> Version("1.2.3a1").pre
        ('a', 1)
        >>> Version("1.2.3b1").pre
        ('b', 1)
        >>> Version("1.2.3rc1").pre
        ('rc', 1)
        """
        return self._version.pre

    @property
    def post(self) -> int | None:
        """The post-release number of the version.

        >>> print(Version("1.2.3").post)
        None
        >>> Version("1.2.3.post1").post
        1
        """
        return self._version.post[1] if self._version.post else None

    @property
    def dev(self) -> int | None:
        """The development number of the version.

        >>> print(Version("1.2.3").dev)
        None
        >>> Version("1.2.3.dev1").dev
        1
        """
        return self._version.dev[1] if self._version.dev else None

    @property
    def local(self) -> str | None:
        """The local version segment of the version.

        >>> print(Version("1.2.3").local)
        None
        >>> Version("1.2.3+abc").local
        'abc'
        """
        if self._version.local:
            return ".".join(str(x) for x in self._version.local)
        else:
            return None

    @property
    def public(self) -> str:
        """The public portion of the version.

        >>> Version("1.2.3").public
        '1.2.3'
        >>> Version("1.2.3+abc").public
        '1.2.3'
        >>> Version("1!1.2.3dev1+abc").public
        '1!1.2.3.dev1'
        """
        return str(self).split("+", 1)[0]

    @property
    def base_version(self) -> str:
        """The "base version" of the version.

        >>> Version("1.2.3").base_version
        '1.2.3'
        >>> Version("1.2.3+abc").base_version
        '1.2.3'
        >>> Version("1!1.2.3dev1+abc").base_version
        '1!1.2.3'

        The "base version" is the public version of the project without any pre or post
        release markers.
        """
        parts = []

        # Epoch
        if self.epoch != 0:
            parts.append(f"{self.epoch}!")

        # Release segment
        parts.append(".".join(str(x) for x in self.release))

        return "".join(parts)

    @property
    def is_prerelease(self) -> bool:
        """Whether this version is a pre-release.

        >>> Version("1.2.3").is_prerelease
        False
        >>> Version("1.2.3a1").is_prerelease
        True
        >>> Version("1.2.3b1").is_prerelease
        True
        >>> Version("1.2.3rc1").is_prerelease
        True
        >>> Version("1.2.3dev1").is_prerelease
        True
        """
        return self.dev is not None or self.pre is not None

    @property
    def is_postrelease(self) -> bool:
        """Whether this version is a post-release.

        >>> Version("1.2.3").is_postrelease
        False
        >>> Version("1.2.3.post1").is_postrelease
        True
        """
        return self.post is not None

    @property
    def is_devrelease(self) -> bool:
        """Whether this version is a development release.

        >>> Version("1.2.3").is_devrelease
        False
        >>> Version("1.2.3.dev1").is_devrelease
        True
        """
        return self.dev is not None

    @property
    def major(self) -> int:
        """The first item of :attr:`release` or ``0`` if unavailable.

        >>> Version("1.2.3").major
        1
        """
        return self.release[0] if len(self.release) >= 1 else 0

    @property
    def minor(self) -> int:
        """The second item of :attr:`release` or ``0`` if unavailable.

        >>> Version("1.2.3").minor
        2
        >>> Version("1").minor
        0
        """
        return self.release[1] if len(self.release) >= 2 else 0

    @property
    def micro(self) -> int:
        """The third item of :attr:`release` or ``0`` if unavailable.

        >>> Version("1.2.3").micro
        3
        >>> Version("1").micro
        0
        """
        return self.release[2] if len(self.release) >= 3 else 0


class _TrimmedRelease(Version):
    @property
    def release(self) -> tuple[int, ...]:
        """
        Release segment without any trailing zeros.

        >>> _TrimmedRelease('1.0.0').release
        (1,)
        >>> _TrimmedRelease('0.0').release
        (0,)
        """
        rel = super().release
        nonzeros = (index for index, val in enumerate(rel) if val)
        last_nonzero = max(nonzeros, default=0)
        return rel[: last_nonzero + 1]


def _parse_letter_version(
    letter: str | None, number: str | bytes | SupportsInt | None
) -> tuple[str, int] | None:
    if letter:
        # We consider there to be an implicit 0 in a pre-release if there is
        # not a numeral associated with it.
        if number is None:
            number = 0

        # We normalize any letters to their lower case form
        letter = letter.lower()

        # We consider some words to be alternate spellings of other words and
        # in those cases we want to normalize the spellings to our preferred
        # spelling.
        if letter == "alpha":
            letter = "a"
        elif letter == "beta":
            letter = "b"
        elif letter in ["c", "pre", "preview"]:
            letter = "rc"
        elif letter in ["rev", "r"]:
            letter = "post"

        return letter, int(number)

    assert not letter
    if number:
        # We assume if we are given a number, but we are not given a letter
        # then this is using the implicit post release syntax (e.g. 1.0-1)
        letter = "post"

        return letter, int(number)

    return None


_local_version_separators = re.compile(r"[\._-]")


def _parse_local_version(local: str | None) -> LocalType | None:
    """
    Takes a string like abc.1.twelve and turns it into ("abc", 1, "twelve").
    """
    if local is not None:
        return tuple(
            part.lower() if not part.isdigit() else int(part)
            for part in _local_version_separators.split(local)
        )
    return None


def _cmpkey(
    epoch: int,
    release: tuple[int, ...],
    pre: tuple[str, int] | None,
    post: tuple[str, int] | None,
    dev: tuple[str, int] | None,
    local: LocalType | None,
) -> CmpKey:
    # When we compare a release version, we want to compare it with all of the
    # trailing zeros removed. So we'll use a reverse the list, drop all the now
    # leading zeros until we come to something non zero, then take the rest
    # re-reverse it back into the correct order and make it a tuple and use
    # that for our sorting key.
    _release = tuple(
        reversed(list(itertools.dropwhile(lambda x: x == 0, reversed(release))))
    )

    # We need to "trick" the sorting algorithm to put 1.0.dev0 before 1.0a0.
    # We'll do this by abusing the pre segment, but we _only_ want to do this
    # if there is not a pre or a post segment. If we have one of those then
    # the normal sorting rules will handle this case correctly.
    if pre is None and post is None and dev is not None:
        _pre: CmpPrePostDevType = NegativeInfinity
    # Versions without a pre-release (except as noted above) should sort after
    # those with one.
    elif pre is None:
        _pre = Infinity
    else:
        _pre = pre

    # Versions without a post segment should sort before those with one.
    if post is None:
        _post: CmpPrePostDevType = NegativeInfinity

    else:
        _post = post

    # Versions without a development segment should sort after those with one.
    if dev is None:
        _dev: CmpPrePostDevType = Infinity

    else:
        _dev = dev

    if local is None:
        # Versions without a local segment should sort before those with one.
        _local: CmpLocalType = NegativeInfinity
    else:
        # Versions with a local segment need that segment parsed to implement
        # the sorting rules in PEP440.
        # - Alpha numeric segments sort before numeric segments
        # - Alpha numeric segments sort lexicographically
        # - Numeric segments sort numerically
        # - Shorter versions sort before longer versions when the prefixes
        #   match exactly
        _local = tuple(
            (i, "") if isinstance(i, int) else (NegativeInfinity, i) for i in local
        )

    return epoch, _release, _pre, _post, _dev, _local
