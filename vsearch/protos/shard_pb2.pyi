from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AddRequest(_message.Message):
    __slots__ = ("global_ids", "dim", "values", "seq")
    GLOBAL_IDS_FIELD_NUMBER: _ClassVar[int]
    DIM_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    SEQ_FIELD_NUMBER: _ClassVar[int]
    global_ids: _containers.RepeatedScalarFieldContainer[int]
    dim: int
    values: _containers.RepeatedScalarFieldContainer[float]
    seq: int
    def __init__(self, global_ids: _Optional[_Iterable[int]] = ..., dim: _Optional[int] = ..., values: _Optional[_Iterable[float]] = ..., seq: _Optional[int] = ...) -> None: ...

class AddResponse(_message.Message):
    __slots__ = ("size",)
    SIZE_FIELD_NUMBER: _ClassVar[int]
    size: int
    def __init__(self, size: _Optional[int] = ...) -> None: ...

class SearchRequest(_message.Message):
    __slots__ = ("query", "k")
    QUERY_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    query: _containers.RepeatedScalarFieldContainer[float]
    k: int
    def __init__(self, query: _Optional[_Iterable[float]] = ..., k: _Optional[int] = ...) -> None: ...

class Hit(_message.Message):
    __slots__ = ("global_id", "distance")
    GLOBAL_ID_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_FIELD_NUMBER: _ClassVar[int]
    global_id: int
    distance: float
    def __init__(self, global_id: _Optional[int] = ..., distance: _Optional[float] = ...) -> None: ...

class SearchResponse(_message.Message):
    __slots__ = ("hits",)
    HITS_FIELD_NUMBER: _ClassVar[int]
    hits: _containers.RepeatedCompositeFieldContainer[Hit]
    def __init__(self, hits: _Optional[_Iterable[_Union[Hit, _Mapping]]] = ...) -> None: ...

class StatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StatusResponse(_message.Message):
    __slots__ = ("size", "durable", "last_seq", "snapshot_seq")
    SIZE_FIELD_NUMBER: _ClassVar[int]
    DURABLE_FIELD_NUMBER: _ClassVar[int]
    LAST_SEQ_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_SEQ_FIELD_NUMBER: _ClassVar[int]
    size: int
    durable: bool
    last_seq: int
    snapshot_seq: int
    def __init__(self, size: _Optional[int] = ..., durable: _Optional[bool] = ..., last_seq: _Optional[int] = ..., snapshot_seq: _Optional[int] = ...) -> None: ...

class FetchLogRequest(_message.Message):
    __slots__ = ("after_seq",)
    AFTER_SEQ_FIELD_NUMBER: _ClassVar[int]
    after_seq: int
    def __init__(self, after_seq: _Optional[int] = ...) -> None: ...

class LogRecord(_message.Message):
    __slots__ = ("seq", "global_ids", "dim", "values")
    SEQ_FIELD_NUMBER: _ClassVar[int]
    GLOBAL_IDS_FIELD_NUMBER: _ClassVar[int]
    DIM_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    seq: int
    global_ids: _containers.RepeatedScalarFieldContainer[int]
    dim: int
    values: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, seq: _Optional[int] = ..., global_ids: _Optional[_Iterable[int]] = ..., dim: _Optional[int] = ..., values: _Optional[_Iterable[float]] = ...) -> None: ...

class FetchSnapshotRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SnapshotChunk(_message.Message):
    __slots__ = ("data", "last_seq")
    DATA_FIELD_NUMBER: _ClassVar[int]
    LAST_SEQ_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    last_seq: int
    def __init__(self, data: _Optional[bytes] = ..., last_seq: _Optional[int] = ...) -> None: ...

class SyncFromRequest(_message.Message):
    __slots__ = ("peer",)
    PEER_FIELD_NUMBER: _ClassVar[int]
    peer: str
    def __init__(self, peer: _Optional[str] = ...) -> None: ...

class SyncFromResponse(_message.Message):
    __slots__ = ("method", "records_applied", "last_seq")
    METHOD_FIELD_NUMBER: _ClassVar[int]
    RECORDS_APPLIED_FIELD_NUMBER: _ClassVar[int]
    LAST_SEQ_FIELD_NUMBER: _ClassVar[int]
    method: str
    records_applied: int
    last_seq: int
    def __init__(self, method: _Optional[str] = ..., records_applied: _Optional[int] = ..., last_seq: _Optional[int] = ...) -> None: ...
