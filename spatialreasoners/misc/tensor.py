from torch import Tensor


def unsqueeze_multi_dims(
    t: Tensor,
    n: int,
    i: int | None = None,
) -> Tensor:
    """
    Arguments:
        t: [d_{0}, ..., d_{n}]
    Returns:
        [d_{0}, ..., d_{i-1}, *(n * (1,)), d_{i}, ..., d_{n}]
    """
    if i is None:
        i = t.ndim
    if i < 0:
        i += t.ndim + 1
        assert i >= 0
    return t[i * (slice(None),) + n * (None,)]


def unsqueeze_as(
    a: Tensor,
    b: Tensor,
    i: int | None = None
) -> Tensor:
    """
    Arguments:
        a: [d_{0}, ..., d_{i}]
        b: [d_{0}, ... d_{n}]
        with i <= n
    Returns:
        a: [d_{0}, ..., d_{i}, *((n-i) * (1,))]
    """
    return unsqueeze_multi_dims(a, b.ndim-a.ndim, i)
