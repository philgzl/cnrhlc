import torch


class Registry:
    """Registry for classes and functions.

    Helper class to register classes and functions in a registry. To register a class or a function to an instance of
    :class:`Registry`, use the :meth:`register` method as a decorator.

    Usage
    -----
    >>> MyRegistry = Registry("registry_name")
    >>> @MyRegistry.register("my_class_key")
    ... class MyClass:
    ...     pass
    >>> my_cls = MyRegistry.get("my_class_key")
    >>> my_obj = my_cls()
    """

    def __init__(self, tag):
        self.tag = tag
        self._registry = {}

    def register(self, key):
        """Register a class or a function."""

        def inner_wrapper(wrapped_class):
            if key in self._registry:
                raise ValueError(f"'{key}' already in {self.tag} registry")
            self._registry[key] = wrapped_class
            return wrapped_class

        return inner_wrapper

    def get(self, key):
        """Get a registered class or function."""
        if key in self._registry:
            return self._registry[key]
        else:
            raise KeyError(f"'{key}' not in {self.tag} registry")

    def keys(self):
        """Get all registered keys."""
        return self._registry.keys()


def impulse(n, idx=0, val=1.0, dtype=torch.float32):
    """Impulse function.

    Equals zero everywhere except at given index.

    Parameters
    ----------
    n : int
        Output length.
    idx : int, optional
        Index of non-zero value.
    val : float, optional
        Non-zero value.
    dtype : torch.dtype, optional
        Output data type.

    Returns
    -------
    torch.Tensor
        Output tensor. Shape ``(n,)``.

    """
    output = torch.zeros(n, dtype=dtype)
    output[idx] = val
    return output


def apply_mask(*args, lengths=None):
    """Set elements of a tensor after given lengths to zero.

    Parameters
    ----------
    args : torch.Tensor
        Input tensors. Shape ``(batch, ..., time)``.
    lengths : torch.Tensor, optional
        Length of tensors along last axis before batching. Shape ``(batch,)``.

    Returns
    -------
    torch.Tensor
        Output tensors with elements after given lengths set to zero.

    """
    if lengths is None:
        return args
    assert len(lengths) == args[0].shape[0]
    mask = torch.zeros(args[0].shape, device=args[0].device)
    for i, length in enumerate(lengths):
        mask[i, ..., :length] = 1
    return (x * mask for x in args)


def linear_interpolation(x, xp, fp):
    """Linear interpolation.

    Parameters
    ----------
    x : torch.Tensor
        Points to interpolate. Shape ``(batch_size, m)`` or ``(m,)``.
    xp : torch.Tensor
        Known x-coordinates. Shape ``(batch_size, n)`` or ``(n,)``.
    fp : torch.Tensor
        Known y-coordinates. Shape ``(batch_size, n)`` or ``(n,)``.

    Returns
    -------
    torch.Tensor
        Interpolated y-coordinates. Shape ``(batch_size, m)`` or ``(m,)``.

    """
    assert x.ndim in [1, 2], x.ndim
    assert xp.ndim in [1, 2], xp.ndim
    assert fp.ndim in [1, 2], fp.ndim
    assert xp.shape == fp.shape, (xp.shape, fp.shape)
    squeeze_output = x.ndim == 1 and xp.ndim == 1
    if x.ndim == 1:
        x = x.unsqueeze(0)
        if xp.ndim == 2:
            x = x.expand(xp.shape[0], -1)
    if xp.ndim == 1:
        xp = xp.unsqueeze(0)
        fp = fp.unsqueeze(0)
        if x.shape[0] == 1:
            xp = xp.expand(x.shape[0], -1)
            fp = fp.expand(x.shape[0], -1)
    indices = torch.searchsorted(xp.contiguous(), x.contiguous()) - 1
    indices = indices.clamp(0, xp.shape[-1] - 2)
    xp_left = xp.gather(-1, indices)
    xp_right = xp.gather(-1, indices + 1)
    fp_left = fp.gather(-1, indices)
    fp_right = fp.gather(-1, indices + 1)
    slope = (fp_right - fp_left) / (xp_right - xp_left)
    output = fp_left + slope * (x - xp_left)
    if squeeze_output:
        assert output.shape[0] == 1
        output = output.squeeze(0)
    return output
