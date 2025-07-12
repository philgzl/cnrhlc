from contextlib import nullcontext

import numpy as np
import pytest
import torch

from cnrhlc.filters import (
    DRNLFilterbank,
    FIRFilterbank,
    GammatoneFilterbank,
    IIRFilterbank,
    butter,
    data_goode1994,
    data_lopezpoveda2001,
    erbspace,
    erbspace_bw,
    gammatone,
    middle_ear_filter,
)
from cnrhlc.utils import impulse


@pytest.mark.parametrize("order", [2])
@pytest.mark.parametrize("fc", [1000])
@pytest.mark.parametrize("fs", [16000])
def test_butter(matlab_engine, order, fc, fs):
    a_python, b_python = butter(order, fc / fs)
    a_matlab, b_matlab = matlab_engine.butter(order, fc / fs, nargout=2)
    a_python = a_python.reshape(1, a_python.shape[-1])
    b_python = b_python.reshape(1, b_python.shape[-1])
    assert a_python.shape == a_matlab.size
    assert b_python.shape == b_matlab.size
    assert np.allclose(a_python, a_matlab)
    assert np.allclose(b_python, b_matlab)


@pytest.mark.parametrize("f_min", [1e2])
@pytest.mark.parametrize("f_max", [1e4])
@pytest.mark.parametrize("n", [30])
def test_erbspace(matlab_engine, f_min, f_max, n):
    f_python = erbspace(f_min, f_max, n)[None, :]
    f_matlab = matlab_engine.erbspace(f_min, f_max, n)
    assert f_python.shape == f_matlab.size
    assert np.allclose(f_python, f_matlab)


@pytest.mark.parametrize("f_min", [1e2])
@pytest.mark.parametrize("f_max", [1e4])
@pytest.mark.parametrize("erb_space", [1.0])
def test_erbspace_bw(matlab_engine, f_min, f_max, erb_space):
    f_python = erbspace_bw(f_min, f_max, erb_space)[None, :]
    f_matlab = matlab_engine.erbspacebw(f_min, f_max, erb_space)
    assert f_python.shape == f_matlab.size
    assert np.allclose(f_python, f_matlab)


@pytest.mark.parametrize("fs", [16e3, 20e3, 32e3])
def test_middle_ear_filter(matlab_engine, fs):
    fir_python = middle_ear_filter(fs)[None, :]
    fir_matlab = matlab_engine.middleearfilter(fs)
    assert fir_python.shape == fir_matlab.size
    assert np.allclose(fir_python, fir_matlab)


@pytest.mark.filterwarnings("ignore:Using iir_output='ba' can lead to numerically ")
@pytest.mark.parametrize("fc", [100, [200], [500, 1000]])
@pytest.mark.parametrize("fs", [16e3, 20e3])
@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("filter_type", ["classic", "allpole"])
def test_gammatone(matlab_engine, fc, fs, order, filter_type):
    matlab = pytest.importorskip("matlab")
    b_python, a_python = gammatone(
        fc,
        fs,
        order=order,
        filter_type=f"amt_{filter_type}",
        iir_output="ba",
    )
    b_matlab, a_matlab = matlab_engine.gammatone(
        matlab.double(fc), fs, float(order), filter_type, nargout=2
    )
    b_python = b_python.reshape(-1, b_python.shape[-1])
    a_python = a_python.reshape(-1, a_python.shape[-1])
    b_matlab = np.array(b_matlab)
    a_matlab = np.array(a_matlab)
    if b_matlab.ndim == 0:
        b_matlab = b_matlab.reshape(1, 1)
    if a_matlab.ndim == 0:
        a_matlab = a_matlab.reshape(1, 1)
    assert b_python.shape == b_matlab.shape
    assert a_python.shape == a_matlab.shape
    assert np.allclose(b_python, b_matlab)
    assert np.allclose(a_python, a_matlab)


def _test_filterbank(torch_rng, input_shape, axis, filter_shape, batching, fb):
    x = torch.randn(input_shape, generator=torch_rng)

    # fill with nans starting at random indices to test causality
    i_shape = [n for i, n in enumerate(input_shape) if i != axis % x.ndim]
    i_nan = torch.randint(0, input_shape[axis], size=i_shape, generator=torch_rng)
    x = x.moveaxis(axis, -1).reshape(-1, x.shape[axis])
    for i, j in enumerate(i_nan.flatten()):
        x[i, j:] = float("nan")
    x = x.reshape(*i_shape, -1).moveaxis(-1, axis)
    assert x.shape == input_shape

    if batching and (
        axis % x.ndim != x.ndim - 1
        or len(input_shape) < 2
        or len(filter_shape) != 2
        or input_shape[-2] != filter_shape[-2]
    ):
        with pytest.raises(ValueError):
            y = fb(x, axis=axis, batching=batching)
        return
    y = fb(x, axis=axis, batching=batching)

    if len(filter_shape) == 1 or batching:
        assert x.shape == y.shape
        y = y.moveaxis(axis, -1)
        y = y.reshape(-1, 1, y.shape[-1])
    else:
        assert x.ndim == y.ndim - 1
        axis = axis % x.ndim
        assert y.shape[axis] == filter_shape[0]
        assert x.shape[:axis] == y.shape[:axis]
        assert x.shape[axis:] == y.shape[axis + 1 :]
        y = y.moveaxis(axis, -1).moveaxis(axis, -1)
        y = y.reshape(-1, y.shape[-2], y.shape[-1])
    for i, j in enumerate(i_nan.flatten()):
        assert torch.isnan(y[i, :, j:]).all()
        assert not torch.isnan(y[i, :, :j]).any()


@pytest.mark.parametrize(
    "input_shape, axis",
    [  # make sure some have input_shape[-2] == filter_shape[-2] to test batching
        [(100,), 0],
        [(100,), -1],
        [(100, 2), 0],
        [(2, 100), 1],
        [(3, 100), -1],
        [(100, 3), -2],
        [(100, 3, 2), 0],
        [(3, 100, 2), 1],
        [(3, 2, 100), 2],
        [(2, 3, 100), -1],
        [(2, 100, 3), -2],
        [(100, 2, 3), -3],
        [(1, 3, 2, 100), -1],
        [(1, 2, 3, 100), 3],
        [(4, 1, 3, 2, 100), -1],
    ],
)
@pytest.mark.parametrize("filter_shape", [(4,), (2, 4)])
@pytest.mark.parametrize("batching", [False, True])
def test_iir_filterbank(torch_rng, input_shape, axis, filter_shape, batching):
    b = torch.randn(filter_shape, generator=torch_rng)
    a = torch.randn(filter_shape, generator=torch_rng)
    fb = IIRFilterbank(b, a)
    _test_filterbank(torch_rng, input_shape, axis, filter_shape, batching, fb)


@pytest.mark.parametrize(
    "input_shape, axis",
    [
        [(100,), 0],
        [(100,), -1],
        [(100, 2), 0],
        [(2, 100), 1],
        [(3, 100), -1],
        [(100, 3), -2],
        [(100, 3, 2), 0],
        [(3, 100, 2), 1],
        [(3, 2, 100), 2],
        [(2, 3, 100), -1],
        [(2, 100, 3), -2],
        [(100, 2, 3), -3],
        [(1, 3, 2, 100), -1],
        [(1, 2, 3, 100), 3],
        [(4, 1, 3, 2, 100), -1],
    ],
)
@pytest.mark.parametrize("filter_shape", [(4,), (2, 4)])
@pytest.mark.parametrize("batching", [False, True])
def test_fir_filterbank(torch_rng, input_shape, axis, filter_shape, batching):
    b = torch.randn(filter_shape, generator=torch_rng)
    fb = FIRFilterbank(b)
    _test_filterbank(torch_rng, input_shape, axis, filter_shape, batching, fb)


@pytest.mark.filterwarnings("ignore:Using iir_output='ba' can lead to numerically ")
@pytest.mark.parametrize("impulse_length", [1000])
@pytest.mark.parametrize("fs", [20e3])
@pytest.mark.parametrize("precision", ["double"])
def test_drnl_filterbank(matlab_engine, impulse_length, fs, precision):
    matlab = pytest.importorskip("matlab")
    dtype = torch.float32 if precision == "single" else torch.float64
    x = impulse(impulse_length, dtype=dtype).requires_grad_()
    fb = DRNLFilterbank(precision=precision)
    y_python = fb(x).T
    y_matlab = matlab_engine.lopezpoveda2001(matlab.double(x.tolist()), fs)
    assert y_python.shape == y_matlab.size
    assert np.allclose(y_python.detach(), y_matlab)
    y_python.mean().backward()


@pytest.mark.parametrize("fs", [60e3])
@pytest.mark.parametrize("f_min", [100])
@pytest.mark.parametrize("f_max", [10000])
@pytest.mark.parametrize("impulse_length", [1.0])
@pytest.mark.parametrize(
    "filter_type",
    ["fir", "gtf", "apgf", "ozgf", "hohmann", "amt_classic", "amt_allpole"],
)
@pytest.mark.parametrize("iir_output", ["ba", "sos"])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_gtfb(fs, f_min, f_max, impulse_length, filter_type, iir_output, precision):
    dtype = torch.float32 if precision == "single" else torch.float64
    x = impulse(round(fs * impulse_length), dtype=dtype)
    is_unstable = iir_output == "ba" and filter_type != "fir"
    with pytest.warns(UserWarning) if is_unstable else nullcontext():
        fb = GammatoneFilterbank(
            fs=fs,
            f_min=f_min,
            f_max=f_max,
            filter_type=filter_type,
            iir_output=iir_output,
            precision=precision,
        )
    with pytest.raises(ValueError) if filter_type == "hohmann" else nullcontext():
        y = fb(x)
    if filter_type == "hohmann":
        return
    if not is_unstable:
        assert not torch.isnan(y).any()
        if filter_type == "amt_allpole":
            assert torch.all(y.abs() < 1e6)
        else:
            assert torch.all(y.abs() < 1)


def test_data_goode1994(matlab_engine):
    data_python = data_goode1994()
    data_matlab = matlab_engine.data_goode1994()
    assert data_python.shape == data_matlab.size
    assert np.allclose(data_python, data_matlab)


def test_data_lopezpoveda2001(matlab_engine):
    data_python = data_lopezpoveda2001()
    data_matlab = matlab_engine.data_lopezpoveda2001("fig2b")
    assert data_python.shape == data_matlab.size
    assert np.allclose(data_python, data_matlab)
