from collections.abc import Sequence
from typing import Any


def to_list(
    elem_or_seq: Any | Sequence[Any], 
    l: int
) -> list[Any]:
    if not isinstance(elem_or_seq, list):
        return l * [elem_or_seq]
    assert len(elem_or_seq) == l
    return list(elem_or_seq)


def to_tuple(
    elem_or_seq: Any | Sequence[Any], 
    l: int
) -> tuple[Any, ...]:
    if not isinstance(elem_or_seq, Sequence):
        return l * (elem_or_seq,)
    assert len(elem_or_seq) == l
    return tuple(elem_or_seq)