from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AddRequest(_message.Message):
    __slots__ = ("global_ids", "dim", "values")
    GLOBAL_IDS_FIELD_NUMBER: _ClassVar[int]
    DIM_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    global_ids: _containers.RepeatedScalarFieldContainer[int]
    dim: int
    values: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, global_ids: _Optional[_Iterable[int]] = ..., dim: _Optional[int] = ..., values: _Optional[_Iterable[float]] = ...) -> None: ...

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
