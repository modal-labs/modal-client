# Copyright Modal Labs 2022

"""This is a vendored copy of parts of the 'psutil' Python package.

It only contains what's needed to run the `net_connections()` function, which
lists active network connections from all processes on the system.
"""


import base64
import errno
import os
import socket
import struct
import sys
from collections import defaultdict, namedtuple

LITTLE_ENDIAN = sys.byteorder == "little"

addr = namedtuple("addr", ["ip", "port"])
sconn = namedtuple("sconn", ["fd", "family", "type", "laddr", "raddr", "status", "pid"])
pconn = namedtuple("pconn", ["fd", "family", "type", "laddr", "raddr", "status"])

# https://github.com/torvalds/linux/blob/master/include/net/tcp_states.h
TCP_STATUSES = {
    "01": "ESTABLISHED",
    "02": "SYN_SENT",
    "03": "SYN_RECV",
    "04": "FIN_WAIT1",
    "05": "FIN_WAIT2",
    "06": "TIME_WAIT",
    "07": "CLOSE",
    "08": "CLOSE_WAIT",
    "09": "LAST_ACK",
    "0A": "LISTEN",
    "0B": "CLOSING",
}


def socktype_to_enum(num):
    """Convert a numeric socket type value to an IntEnum member.
    If it's not a known member, return the numeric value itself.
    """
    try:
        return socket.SocketKind(num)
    except ValueError:
        return num


# The read buffer size for open() builtin. This (also) dictates how
# much data we read(2) when iterating over file lines as in:
#   >>> with open(file) as f:
#   ...    for line in f:
#   ...        ...
# Default per-line buffer size for binary files is 1K. For text files
# is 8K. We use a bigger buffer (32K) in order to have more consistent
# results when reading /proc pseudo files on Linux, see:
# https://github.com/giampaolo/psutil/issues/2050
# On Python 2 this also speeds up the reading of big files:
# (namely /proc/{pid}/smaps and /proc/net/*):
# https://github.com/giampaolo/psutil/issues/708
FILE_READ_BUFFER_SIZE = 32 * 1024

ENCODING = sys.getfilesystemencoding()
ENCODING_ERRS = sys.getfilesystemencodeerrors()


def open_text(fname):
    """On Python 3 opens a file in text mode by using fs encoding and
    a proper en/decoding errors handler.
    On Python 2 this is just an alias for open(name, 'rt').
    """

    # See:
    # https://github.com/giampaolo/psutil/issues/675
    # https://github.com/giampaolo/psutil/pull/733
    fobj = open(
        fname,
        buffering=FILE_READ_BUFFER_SIZE,
        encoding=ENCODING,
        errors=ENCODING_ERRS,
    )
    try:
        # Dictates per-line read(2) buffer size. Defaults is 8k. See:
        # https://github.com/giampaolo/psutil/issues/2050#issuecomment-1013387546
        fobj._CHUNK_SIZE = FILE_READ_BUFFER_SIZE
    except AttributeError:
        pass
    except Exception:
        fobj.close()
        raise

    return fobj


def path_exists_strict(path):
    """Same as os.path.exists() but does not swallow EACCES / EPERM
    exceptions. See:
    http://mail.python.org/pipermail/python-dev/2012-June/120787.html.
    """
    try:
        os.stat(path)
    except OSError as err:
        if err.errno in (errno.EPERM, errno.EACCES):
            raise
        return False
    else:
        return True


def readlink(path):
    """Wrapper around os.readlink()."""
    assert isinstance(path, str), path
    path = os.readlink(path)
    # readlink() might return paths containing null bytes ('\x00')
    # resulting in "TypeError: must be encoded string without NULL
    # bytes, not str" errors when the string is passed to other
    # fs-related functions (os.*, open(), ...).
    # Apparently everything after '\x00' is garbage (we can have
    # ' (deleted)', 'new' and possibly others), see:
    # https://github.com/giampaolo/psutil/issues/717
    path = path.split("\x00")[0]
    # Certain paths have ' (deleted)' appended. Usually this is
    # bogus as the file actually exists. Even if it doesn't we
    # don't care.
    if path.endswith(" (deleted)") and not path_exists_strict(path):
        path = path[:-10]
    return path


def pids():
    """Returns a list of PIDs currently running on the system."""
    return [int(x) for x in os.listdir("/proc") if x.isdigit()]


class Connections:
    """A wrapper on top of /proc/net/* files, retrieving per-process
    and system-wide open connections (TCP, UDP, UNIX) similarly to
    "netstat -an".

    Note: in case of UNIX sockets we're only able to determine the
    local endpoint/path, not the one it's connected to.
    According to [1] it would be possible but not easily.

    [1] http://serverfault.com/a/417946
    """

    def __init__(self):
        # The string represents the basename of the corresponding
        # /proc/net/{proto_name} file.
        tcp4 = ("tcp", socket.AF_INET, socket.SOCK_STREAM)
        tcp6 = ("tcp6", socket.AF_INET6, socket.SOCK_STREAM)
        udp4 = ("udp", socket.AF_INET, socket.SOCK_DGRAM)
        udp6 = ("udp6", socket.AF_INET6, socket.SOCK_DGRAM)
        unix = ("unix", socket.AF_UNIX, None)
        self.tmap = {
            "all": (tcp4, tcp6, udp4, udp6, unix),
            "tcp": (tcp4, tcp6),
            "tcp4": (tcp4,),
            "tcp6": (tcp6,),
            "udp": (udp4, udp6),
            "udp4": (udp4,),
            "udp6": (udp6,),
            "unix": (unix,),
            "inet": (tcp4, tcp6, udp4, udp6),
            "inet4": (tcp4, udp4),
            "inet6": (tcp6, udp6),
        }

    def get_proc_inodes(self, pid):
        inodes = defaultdict(list)
        for fd in os.listdir(f"/proc/{pid}/fd"):
            try:
                inode = readlink(f"/proc/{pid}/fd/{fd}")
            except (FileNotFoundError, ProcessLookupError):
                # ENOENT == file which is gone in the meantime;
                # os.stat('/proc/%s' % self.pid) will be done later
                # to force NSP (if it's the case)
                continue
            except OSError as err:
                if err.errno == errno.EINVAL:
                    # not a link
                    continue
                if err.errno == errno.ENAMETOOLONG:
                    # file name too long
                    continue
                raise
            else:
                if inode.startswith("socket:["):
                    # the process is using a socket
                    inode = inode[8:][:-1]
                    inodes[inode].append((pid, int(fd)))
        return inodes

    def get_all_inodes(self):
        inodes = {}
        for pid in pids():
            try:
                inodes.update(self.get_proc_inodes(pid))
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                # os.listdir() is gonna raise a lot of access denied
                # exceptions in case of unprivileged user; that's fine
                # as we'll just end up returning a connection with PID
                # and fd set to None anyway.
                # Both netstat -an and lsof does the same so it's
                # unlikely we can do any better.
                # ENOENT just means a PID disappeared on us.
                continue
        return inodes

    @staticmethod
    def decode_address(addr_, family):
        """Accept an "ip:port" address as displayed in /proc/net/*
        and convert it into a human readable form, like:

        "0500000A:0016" -> ("10.0.0.5", 22)
        "0000000000000000FFFF00000100007F:9E49" -> ("::ffff:127.0.0.1", 40521)

        The IP address portion is a little or big endian four-byte
        hexadecimal number; that is, the least significant byte is listed
        first, so we need to reverse the order of the bytes to convert it
        to an IP address.
        The port is represented as a two-byte hexadecimal number.

        Reference:
        http://linuxdevcenter.com/pub/a/linux/2000/11/16/LinuxAdmin.html
        """
        ip, port = addr_.split(":")
        port = int(port, 16)
        # this usually refers to a local socket in listen mode with
        # no end-points connected
        if not port:
            return ()
        ip = ip.encode("ascii")
        if family == socket.AF_INET:
            # see: https://github.com/giampaolo/psutil/issues/201
            if LITTLE_ENDIAN:
                ip = socket.inet_ntop(family, base64.b16decode(ip)[::-1])
            else:
                ip = socket.inet_ntop(family, base64.b16decode(ip))
        else:  # IPv6
            ip = base64.b16decode(ip)
            if LITTLE_ENDIAN:
                ip = socket.inet_ntop(
                    socket.AF_INET6,
                    struct.pack(">4I", *struct.unpack("<4I", ip)),
                )
            else:
                ip = socket.inet_ntop(
                    socket.AF_INET6,
                    struct.pack("<4I", *struct.unpack("<4I", ip)),
                )
        return addr(ip, port)

    @staticmethod
    def process_inet(file, family, type_, inodes, filter_pid=None):
        """Parse /proc/net/tcp* and /proc/net/udp* files."""
        if file.endswith("6") and not os.path.exists(file):
            # IPv6 not supported
            return
        with open_text(file) as f:
            f.readline()  # skip the first line
            for lineno, line in enumerate(f, 1):
                try:
                    _, laddr, raddr, status, _, _, _, _, _, inode = line.split()[:10]
                except ValueError:
                    raise RuntimeError("error while parsing %s; malformed line %s %r" % (file, lineno, line))
                if inode in inodes:
                    # # We assume inet sockets are unique, so we error
                    # # out if there are multiple references to the
                    # # same inode. We won't do this for UNIX sockets.
                    # if len(inodes[inode]) > 1 and family != socket.AF_UNIX:
                    #     raise ValueError("ambiguous inode with multiple "
                    #                      "PIDs references")
                    pid, fd = inodes[inode][0]
                else:
                    pid, fd = None, -1
                if filter_pid is not None and filter_pid != pid:
                    continue
                else:
                    if type_ == socket.SOCK_STREAM:
                        status = TCP_STATUSES[status]
                    else:
                        status = "NONE"
                    laddr = Connections.decode_address(laddr, family)
                    raddr = Connections.decode_address(raddr, family)
                    yield (fd, family, type_, laddr, raddr, status, pid)

    @staticmethod
    def process_unix(file, family, inodes, filter_pid=None):
        """Parse /proc/net/unix files."""
        with open_text(file) as f:
            f.readline()  # skip the first line
            for line in f:
                tokens = line.split()
                try:
                    _, _, _, _, type_, _, inode = tokens[0:7]
                except ValueError:
                    if " " not in line:
                        # see: https://github.com/giampaolo/psutil/issues/766
                        continue
                    raise RuntimeError("error while parsing %s; malformed line %r" % (file, line))
                if inode in inodes:  # noqa
                    # With UNIX sockets we can have a single inode
                    # referencing many file descriptors.
                    pairs = inodes[inode]
                else:
                    pairs = [(None, -1)]
                for pid, fd in pairs:
                    if filter_pid is not None and filter_pid != pid:
                        continue
                    else:
                        path = tokens[-1] if len(tokens) == 8 else ""
                        type_ = socktype_to_enum(int(type_))
                        # XXX: determining the remote endpoint of a
                        # UNIX socket on Linux is not possible, see:
                        # https://serverfault.com/questions/252723/
                        raddr = ""
                        status = "NONE"
                        yield (fd, family, type_, path, raddr, status, pid)

    def retrieve(self, kind, pid=None):
        if kind not in self.tmap:
            raise ValueError(
                "invalid %r kind argument; choose between %s" % (kind, ", ".join([repr(x) for x in self.tmap]))
            )
        if pid is not None:
            inodes = self.get_proc_inodes(pid)
            if not inodes:
                # no connections for this process
                return []
        else:
            inodes = self.get_all_inodes()
        ret = set()
        for proto_name, family, type_ in self.tmap[kind]:
            path = f"/proc/net/{proto_name}"
            if family in (socket.AF_INET, socket.AF_INET6):
                ls = self.process_inet(path, family, type_, inodes, filter_pid=pid)
            else:
                ls = self.process_unix(path, family, inodes, filter_pid=pid)
            for fd, family, type_, laddr, raddr, status, bound_pid in ls:
                if pid:
                    conn = pconn(fd, family, type_, laddr, raddr, status)
                else:
                    conn = sconn(fd, family, type_, laddr, raddr, status, bound_pid)
                ret.add(conn)
        return list(ret)


_connections = Connections()


def net_connections(kind="inet"):
    """Return system-wide open connections."""
    return _connections.retrieve(kind)
