import pytest
import sys

import grpclib.events
from grpclib import Status

from modal._utils.grpclib_patch import (
    PatchedRecvInitialMetadata,
    PatchedRecvMessage,
    PatchedRecvRequest,
    PatchedRecvTrailingMetadata,
    PatchedSendInitialMetadata,
    PatchedSendMessage,
    PatchedSendRequest,
    PatchedSendTrailingMetadata,
)

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 14),
    reason="grpclib is patched only in Python 3.14+",
)


@pytest.mark.parametrize(
    "PatchedEvent, GrpclibEvent",
    (
        (PatchedSendMessage, grpclib.events.SendMessage),
        (PatchedRecvMessage, grpclib.events.RecvMessage),
        (PatchedRecvRequest, grpclib.events.RecvRequest),
        (PatchedSendRequest, grpclib.events.SendRequest),
        (PatchedRecvInitialMetadata, grpclib.events.RecvInitialMetadata),
        (PatchedRecvTrailingMetadata, grpclib.events.RecvTrailingMetadata),
        (PatchedSendInitialMetadata, grpclib.events.SendInitialMetadata),
        (PatchedSendTrailingMetadata, grpclib.events.SendTrailingMetadata),
    ),
)
def test_dunder_attributes_set_correctly(PatchedEvent, GrpclibEvent):
    from inspect import get_annotations

    annotations = get_annotations(GrpclibEvent)
    expected_slots = set(name for name in annotations)
    assert PatchedEvent.__payload__ == GrpclibEvent.__payload__

    payload = GrpclibEvent.__payload__
    readonly = frozenset(name for name in annotations if name not in payload)
    assert set(PatchedEvent.__slots__) == expected_slots
    assert readonly == PatchedEvent.__readonly__


def test_readonly_works():
    multidict = pytest.importorskip("multidict")
    trailing_metadata = PatchedSendTrailingMetadata(
        metadata=multidict.MultiDict({"a": "b"}),
        status=Status.OK,
        status_message=None,
        status_details=None,
    )

    with pytest.raises(AttributeError, match="Read-only"):
        trailing_metadata.status = Status.NOT_FOUND

    new_metadata = multidict.MultiDict({"x": "y"})
    trailing_metadata.metadata = new_metadata
    assert trailing_metadata.metadata == new_metadata


def test_interrupted_works():
    multidict = pytest.importorskip("multidict")
    event = PatchedRecvInitialMetadata(metadata=multidict.MultiDict({"a": "b"}))

    assert not event.__interrupted__
    event.interrupt()
    assert event.__interrupted__
