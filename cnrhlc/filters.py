import math
import warnings

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from numpy.polynomial import polynomial as P

from .utils import Registry, impulse

FilterbankRegistry = Registry("filterbank")
"""Registry for filterbanks.

Members of this registry are:

.. list-table::
    :header-rows: 1

    * - Key
      - Member
    * - ``"gammatone"``
      - :class:`GammatoneFilterbank`
    * - ``"drnl"``
      - :class:`DRNLFilterbank`
    * - ``"none"``
      - :class:`NoFilterbank`

"""


def freq_to_erb(freq):
    """Convert frequency to ERB-number.

    Same as ``freqtoerb.m`` in the AMT.
    """
    return 9.2645 * np.sign(freq) * np.log(1 + np.abs(freq) * 0.00437)


def erb_to_freq(erb):
    """Convert ERB-number to frequency.

    Same as ``erbtofreq.m`` in the AMT.
    """
    return (1.0 / 0.00437) * np.sign(erb) * (np.exp(np.abs(erb) / 9.2645) - 1)


def erbspace(f_min, f_max, n):
    """Create an array of frequencies evenly spaced on a ERB-number scale.

    Same as ``erbspace.m`` in the AMT.
    """
    erb_min, erb_max = freq_to_erb(f_min), freq_to_erb(f_max)
    erbs = np.linspace(erb_min, erb_max, n)
    return erb_to_freq(erbs)


def erbspace_bw(f_min, f_max, erb_space):
    """Create an array of frequencies with given ERB-number spacing.

    Same as ``erbspacebw.m`` in the AMT.
    """
    erb_min, erb_max = freq_to_erb(f_min), freq_to_erb(f_max)
    erb_range = erb_max - erb_min
    n = erb_range // erb_space
    remainder = erb_range - n * erb_space
    erbs = erb_min + np.arange(n + 1) * erb_space + remainder / 2
    return erb_to_freq(erbs)


def aud_filt_bw(fc):
    """Critical bandwidth of auditory filter at given center frequency.

    Same as ``audfiltbw.m`` in the AMT.
    """
    return 24.7 + fc / 9.265


def fir2(order, freq, amp):
    """FIR filter design.

    Same as ``fir2.m`` in the Signal Processing Toolbox. ``fir2.m`` uses an
    interpolation method that is not equivalent to :func:`numpy.interp` used inside
    :func:`scipy.signal.firwin2`.

    Parameters
    ----------
    order : int
        Filter order.
    freq : list or numpy.ndarray
        Frequency values at which the amplitude response is sampled.
    amp : list or numpy.ndarray
        Amplitude response values. Same shape as ``freq``.

    Returns
    -------
    numpy.ndarray
        Filter coefficients. Shape ``(order + 1,)``.

    """
    if isinstance(freq, list):
        freq = np.array(freq)
    if isinstance(amp, list):
        amp = np.array(amp)
    if freq.ndim != 1 or amp.ndim != 1:
        raise ValueError("freq and amp must be one-dimensional")
    if len(freq) != len(amp):
        raise ValueError("freq and amp must have the same length")
    if freq[0] != 0.0 or freq[-1] != 1.0:
        raise ValueError("freq must start with 0.0 and end with 1.0")
    d = np.diff(freq)
    if (d <= 0).any():
        raise ValueError("freq must be strictly increasing")
    if order % 2 == 1 and amp[-1] == 0:
        raise ValueError("odd order filter requires zero gain at Nyquist frequency")

    # fir2.m forces the minimum number of interpolation points to 513
    n_taps = order + 1
    if n_taps < 1024:
        n_interp = 513
    else:
        n_interp = 2 * math.ceil(math.log2(n_taps)) + 1

    # Interpolation
    H = np.zeros(n_interp)
    i_start = 0
    for i in range(len(freq) - 1):
        i_end = math.floor(freq[i + 1] * n_interp)
        inc = np.arange(i_end - i_start) / (i_end - i_start - 1)
        H[i_start:i_end] = inc * amp[i + 1] + (1 - inc) * amp[i]
        i_start = i_end

    # Phase shift
    shift = np.exp(-0.5 * (n_taps - 1) * 1j * np.pi * np.linspace(0, 1, n_interp))
    fir = np.fft.irfft(H * shift)

    # Keep first n_taps coefficients and multiply by window
    return fir[:n_taps] * np.hamming(n_taps)


def middle_ear_filter(fs, order=512, min_phase=True):
    """Create a middle ear FIR filter.

    Same as ``middleearfilter.m`` with argument ``lopezpoveda2001`` in the AMT.
    """
    data = data_lopezpoveda2001()
    if fs <= 20000:
        data = data[data[:, 0] < fs / 2, :]
    else:
        i = np.arange(1, 1 + (fs / 2 - data[-1, 0]) // 1000).reshape(-1, 1)
        data = np.vstack(
            [
                data,
                np.hstack(
                    [
                        data[-1, 0] + i * 1000,
                        data[-1, 1] / 1.1**i,
                    ]
                ),
            ]
        )
    if data[-1, 0] != fs / 2:
        data = np.vstack(
            [
                data,
                np.array([[fs / 2, data[-1, 1] / (1 + (fs / 2 - data[-1, 0]) * 1e-4)]]),
            ]
        )
    data = np.vstack([np.array([0, 0]), data * np.array([[2 / fs, 1]])])
    b = fir2(order, data[:, 0], data[:, 1])
    b = b / 20e-6
    if min_phase:
        b = np.fft.fft(b)
        b = np.abs(b) * np.exp(-1j * scipy.signal.hilbert(np.log(np.abs(b))).imag)
        b = np.fft.ifft(b).real
    return b


def butter(order, fc, **kwargs):
    """Butterworth filter design.

    A wrapper around :func:`scipy.signal.butter` with support for multiple cutoff
    frequencies.

    Parameters
    ----------
    order : int
        Filter order.
    fc : float or list or numpy.ndarray
        Cutoff frequency. If :class:`numpy.ndarray`, must have shape ``(n_filters,)``.
    kwargs : dict, optional
        Additional keyword arguments passed to :func:`scipy.signal.butter`.

    Returns
    -------
    b : numpy.ndarray
        Numerator coefficients. Shape ``(n_taps,)`` or ``(n_filters, n_taps)``.
    a : numpy.ndarray
        Denominator coefficients. Same shape as ``b``.

    """
    if "output" in kwargs and kwargs["output"] != "ba":
        raise ValueError(f"only 'ba' output is supported, got {kwargs['output']}")

    fc, scalar_input = _check_0d_or_1d(fc, "fc")

    b = np.empty((len(fc), order + 1))
    a = np.empty((len(fc), order + 1))
    for i, f in enumerate(fc):
        b[i, :], a[i, :] = scipy.signal.butter(order, f, **kwargs)

    if scalar_input:
        b, a = b[0, :], a[0, :]
    return b, a


def matched_z_transform(
    poles, zeros=None, fs=1.0, gain=None, f0db=None, complex=False, _z_zeros=None
):
    """Analog to digital filter using the matched Z-transform method.

    See https://en.wikipedia.org/wiki/Matched_Z-transform_method.

    Parameters
    ----------
    poles : list or numpy.ndarray
        Poles in the s-plane. Shape ``(n_poles,)`` or ``(n_filters, n_poles)``.
    zeros : list or numpy.ndarray, optional
        Zeros in the s-plane. Shape ``(n_zeros,)`` or ``(n_filters, n_zeros)``. If
        ``None``, the filter is all-pole.
    fs : float, optional
        Sampling frequency.
    gain : float or list or numpy.ndarray, optional
        Continuous filter gain. If :class:`numpy.ndarray`, must have shape
        ``(n_filters,)``. If ``None`` and ``f0db`` is also ``None``, no gain is applied
        and the first coefficient in ``b`` is ``1.0``.
    f0db : float, optional
        Frequency at which the filter should have unit gain.  If ``None`` and ``gain``
        is also ``None``, no gain is applied and the first coefficient in ``b`` is
        ``1.0``.
    complex : bool, optional
        If ``False``, the imaginary part of the filter coefficients is discarded and the
        output is real. This is usually desired if the poles and zeros are conjugate,
        since the imaginary part should be zero and the filter can be implemented in the
        real domain. If ``True``, the output coefficients are complex.

    Returns
    -------
    b : numpy.ndarray
        Numerator coefficients. Shape ``(n_filters, n_zeros + 1)`` or
        ``(n_zeros + 1,)``.
    a : numpy.ndarray
        Denominator coefficients. Shape ``(n_filters, n_poles + 1)`` or
        ``(n_poles + 1,)``.

    """
    poles, poles_was_1d = _check_1d_or_2d(poles, "poles")
    if zeros is None:
        zeros, zeros_was_1d = np.empty((poles.shape[0], 0)), poles_was_1d
    else:
        zeros, zeros_was_1d = _check_1d_or_2d(zeros, "zeros")

    both_1d = poles_was_1d and zeros_was_1d
    both_2d = not poles_was_1d and not zeros_was_1d
    if not both_1d and not both_2d:
        raise ValueError("poles and zeros must have the same number of dimensions")
    if both_2d and poles.shape[0] != zeros.shape[0]:
        raise ValueError("poles and zeros must have the same shape along first axis")

    z_poles = np.exp(poles / fs)
    # _z_zeros is used for hard-setting the Z-domain zeros which is useful for the
    # amt_classic and amt_allpole gammatone filters. It should not be used in general!
    z_zeros = np.exp(zeros / fs) if _z_zeros is None else _z_zeros

    # TODO: find a way to vectorize P.polyfromroots instead of lopping over b and a
    b = np.empty((z_zeros.shape[0], z_zeros.shape[1] + 1), dtype="complex")
    a = np.empty((z_poles.shape[0], z_poles.shape[1] + 1), dtype="complex")
    for i in range(z_poles.shape[0]):
        b[i, ::-1] = P.polyfromroots(z_zeros[i, :])
        a[i, ::-1] = P.polyfromroots(z_poles[i, :])
    if not complex:
        b, a = b.real, a.real

    if gain is not None and f0db is not None:
        raise ValueError("cannot specify both gain and f0db")
    elif gain is not None:
        # _z_zeros cannot be used together with s-domain gain since calculating the
        # corresponding Z-domain gain requires the initial s-domain zeros.
        if _z_zeros is not None:
            raise ValueError("cannot specify both gain and _z_zeros")
        gain, _ = _check_0d_or_1d(gain)
        z_gain = np.abs(
            gain
            * np.prod(-zeros, axis=1)
            / np.prod(-poles, axis=1)
            * np.prod(1 - z_poles, axis=1)
            / np.prod(1 - z_zeros, axis=1)
        )
        b = z_gain[:, None] * b
    elif f0db is not None:
        f0db, _ = _check_0d_or_1d(f0db)
        z_f0db = np.exp(-1j * 2 * np.pi * f0db / fs)
        z_gain = np.abs(
            np.prod(1 - z_poles * z_f0db[:, None], axis=1)
            / np.prod(1 - z_zeros * z_f0db[:, None], axis=1)
        )
        b = z_gain[:, None] * b

    if both_1d:
        b, a = b[0, :], a[0, :]
    return b, a


def gammatone(
    fc,
    fs,
    order=4,
    bw_mult=None,
    filter_type="apgf",
    fir_ntaps=512,
    fir_dur=None,
    iir_output="sos",
    compensate_delay=False,
):
    """Gammatone filter coefficients.

    Parameters
    ----------
    fc : float or list or numpy.ndarray
        Center frequency. If :class:`numpy.ndarray`, must have shape ``(n_filters,)``.
    fs : float
        Sampling frequency.
    order : int, optional
        Filter order.
    bw_mult : float or numpy.ndarray, optional
        Bandwidth scaling factor. If :class:`numpy.ndarray`, must have shape
        ``(n_filters,)``. If ``None``, the formula from [1] is used.
    filter_type : str, optional
        Filter type:

        - ``"fir"``: A FIR filter is created by evaluating the time-domain expression of
          the gammatone filter over a finite window. The number of taps is
          ``fir_ntaps``.
        - ``"gtf"``: Accurate IIR equivalent by numerically calculating the s-plane
          zeros.
        - ``"apgf"``: All-pole approximation from [2].
        - ``"ozgf"``: One-zero approximation from [2]. The zero is set to 0 which
          matches the DAPGF denomination in more recent papers by Lyon.
        - ``"hohmann"``: Complex-valued all-pole filter as in [3].
        - ``"amt_classic"``: Mixed pole-zero approximation from [?]. Matches the
          ``'classic'`` option in the AMT.
        - ``"amt_allpole"``: Same as ``"apgf"`` but uses a different scaling. Matches
          the ``'allpole'`` option in the AMT.
    fir_ntaps : int, optional
        Number of taps for the FIR filter. Ignored if ``filter_type != "fir"``.
    fir_dur : float, optional
        Duration of FIR in seconds. Ignored if ``filter_type != "fir"``. Specify either
        ``fir_ntaps`` or ``fir_dur``, not both.
    iir_output : {"ba", "sos"}, optional
        Output format for IIR filters. Either a sequence of ``b`` and ``a`` coefficients
        (``"ba"``), or a sequence of second-order sections (``"sos"``). For stability,
        ``"sos"`` is recommended, but is computationally more expensive. Ignored if
        ``filter_type == "fir"``.
    compensate_delay : bool, optional
        Whether to compensate for the delay introduced by the filters.

    Returns
    -------
    fir : numpy.ndarray
        FIR filter coefficients. Shape ``(n_taps,)`` or ``(n_filters, n_taps)``. Only
        returned if ``filter_type == "fir"``.
    b, a : numpy.ndarray, numpy.ndarray
        Numerator and denominator coefficients. Shape ``(n_taps,)`` or
        ``(n_filters, n_taps)``. Only returned if ``filter_type != "fir"`` and
        ``iir_output == "ba"``.
    sos : numpy.ndarray
        Second-order sections. Shape ``(order, 6)`` or ``(n_filters, order, 6)``. Only
        returned if ``filter_type != "fir"`` and ``iir_output == "sos"``.


    .. [1] J. Holdsworth, I. Nimmo-Smith, R. D. Patterson and P. Rice, "Annex C of the
       SVOS final report: Implementing a gammatone filter bank", Annex C of APU report
       2341, 1988.
    .. [2] R. F. Lyon, "The all-pole gammatone filter and auditory models", in Proc.
       Forum Acusticum, 1996.
    .. [3] V. Hohmann, "Frequency analysis and synthesis using a Gammatone filterbank",
       in Acta Acust. United Acous., 2002.

    """
    fc, scalar_input = _check_0d_or_1d(fc, "fc")

    if bw_mult is None:
        bw_mult = _bw_mult_from_order(order)

    bw = 2 * np.pi * bw_mult * aud_filt_bw(fc)
    wc = 2 * np.pi * fc

    if filter_type == "fir":
        if (fir_ntaps is not None) and (fir_dur is not None):
            msg = f"{fir_ntaps=} conflicts with {fir_dur=} and {fs=}"
            assert fir_ntaps == int(fir_dur * fs), msg
        if fir_ntaps is None:
            fir_ntaps = int(fir_dur * fs)
        t = np.arange(fir_ntaps) / fs
        a = (
            2
            / math.factorial(order - 1)
            / np.abs(1 / bw**order + 1 / (bw + 2j * wc) ** order)
            / fs
        )
        phi = (order - 1) / bw * wc if compensate_delay else np.zeros_like(fc)
        fir = (
            a[:, None]
            * t ** (order - 1)
            * np.exp(-bw[:, None] * t[None, :])
            * np.cos(wc[:, None] * t[None, :] - phi[:, None])
        )
        outputs = (fir,)
    else:
        pole = -bw + 1j * wc
        poles = np.stack([pole, pole.conj()], axis=1)
        zeros = None
        _z_zeros = None
        complex = False
        if filter_type == "gtf":
            zeros = np.zeros((len(fc), order))
            for i in range(len(fc)):
                zeros[i, :] = P.polyroots(
                    P.polyadd(
                        P.polypow([-pole[i], 1], order),
                        P.polypow([-pole[i].conj(), 1], order),
                    )
                ).real
        elif filter_type == "apgf":
            pass
        elif filter_type == "ozgf":
            zeros = np.zeros((len(fc), 1))
        elif filter_type in ["amt_classic", "amt_allpole"]:
            # The MATLAB code sets the Z-domain zeros to the real part of the Z-domain
            # poles, which is wrong! Moreover, those zeros are used for calculating the
            # gain for both classic and allpole, which is probably why a warning is
            # raised about the scaling being wrong for allpole!
            _z_zeros = np.stack(order * [np.exp((pole) / fs).real], axis=1)
        elif filter_type == "hohmann":
            # override poles and complex
            poles = np.stack([pole], axis=1)
            complex = True
        else:
            raise ValueError(f"invalid filter_type, got {filter_type}")
        if iir_output == "ba":
            warnings.warn(
                "Using iir_output='ba' can lead to numerically unstable gammatone "
                "filters. Consider using filter_type='fir' or iir_output='sos' instead."
            )
            poles = np.tile(poles, (1, order))
            b, a = matched_z_transform(
                poles,
                zeros,
                fs=fs,
                f0db=fc,
                complex=complex,
                _z_zeros=_z_zeros,
            )
            if filter_type == "amt_allpole":
                b = b[:, :1]
            outputs = b, a
        elif iir_output == "sos":
            sos = np.empty((len(fc), order, 6), dtype="complex" if complex else "float")
            for i in range(order):
                zeros_i = None if zeros is None else zeros[:, i : i + 1]
                _z_zeros_i = None if _z_zeros is None else _z_zeros[:, i : i + 1]
                b, a = matched_z_transform(
                    poles, zeros_i, fs=fs, f0db=fc, complex=complex, _z_zeros=_z_zeros_i
                )
                if filter_type == "amt_allpole":
                    b = b[:, :1]
                b = np.hstack([b, np.zeros((len(fc), 3 - b.shape[-1]))])
                a = np.hstack([a, np.zeros((len(fc), 3 - a.shape[-1]))])
                sos[:, i, :] = np.hstack([b, a])
            outputs = (sos,)
        else:
            raise ValueError(f"iir_output must be 'ba' or 'sos', got '{iir_output}'")

    if scalar_input:
        outputs = tuple(x[0, ...] for x in outputs)
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs


class FIRFilterbank(nn.Module):
    """FIR filterbank.

    Parameters
    ----------
    fir : list or numpy.ndarray or torch.Tensor
        Filter coefficients. Shape ``(n_taps,)`` or ``(n_filters, n_taps)``.
    dtype : torch.dtype, optional
        Data type to cast ``fir`` to in case it is not a ``torch.Tensor``.

    """

    def __init__(self, fir, dtype=torch.float32):
        super().__init__()
        if not isinstance(fir, (list, np.ndarray, torch.Tensor)):
            raise TypeError(
                "fir must be list, np.ndarray or torch.Tensor, got "
                f"{fir.__class__.__name__}"
            )
        if isinstance(fir, (list, np.ndarray)):
            fir = torch.tensor(fir, dtype=dtype)
        if fir.ndim not in [1, 2]:
            raise ValueError(
                "fir must be one- or two-dimensional with shape (n_taps,) or "
                f"(n_filters, n_taps), got shape {fir.shape}"
            )
        self.register_buffer("fir", fir)

    def forward(self, x, axis=-1, batching=False):
        """Filter input signal.

        Parameters
        ----------
        x : torch.Tensor
            Input signal.
        axis : int, optional
            Axis along which to filter.
        batching : bool, optional
            If ``True``, the input is assumed to have shape ``(..., n_filters, time)``
            and each channel is filtered with its own filter.

        Returns
        -------
        torch.Tensor
            Filtered signal.

        """
        axis = axis % x.ndim
        if batching:
            _batching_check(x, axis, self.fir)
        x = x.moveaxis(axis, -1)
        y = x.reshape(-1, x.shape[-2] if batching else 1, x.shape[-1])
        y = F.conv1d(
            F.pad(y, (self.fir.shape[-1] - 1, 0)),
            self.fir.flip(-1).view(-1, 1, self.fir.shape[-1]),
            groups=y.shape[-2] if batching else 1,
        )
        if self.fir.ndim == 1 or batching:
            y = y.squeeze(-2).reshape(x.shape).moveaxis(-1, axis)
        else:
            y = y.reshape(*x.shape[:-1], -1, x.shape[-1])
            y = y.moveaxis(-1, axis).moveaxis(-1, axis)
        return y


class IIRFilterbank(nn.Module):
    """IIR filterbank.

    Parameters
    ----------
    b : list or numpy.ndarray or torch.Tensor
        Numerator coefficients. Shape ``(n_taps,)`` or ``(n_filters, n_taps)``. If
        shorter than ``a`` along the last axis, it is padded with zeros.
    a : list or numpy.ndarray or torch.Tensor
        Denominator coefficients. Shape ``(n_taps,)`` or ``(n_filters, n_taps)``. If
        shorter than ``b`` along the last axis, it is padded with zeros.
    dtype : torch.dtype, optional
        Data type to cast ``a`` and ``b`` to in case they are not ``torch.Tensor``.

    """

    def __init__(self, b, a, dtype=torch.float32):
        super().__init__()
        if not isinstance(b, (list, np.ndarray, torch.Tensor)) or not isinstance(
            a, (list, np.ndarray, torch.Tensor)
        ):
            raise TypeError(
                "b and a must be list, np.ndarray or torch.Tensor, got "
                f"{b.__class__.__name__} and {a.__class__.__name__}"
            )
        if isinstance(b, (list, np.ndarray)):
            b = torch.tensor(b, dtype=dtype)
        if isinstance(a, (list, np.ndarray)):
            a = torch.tensor(a, dtype=dtype)
        if a.ndim == b.ndim == 1 or a.ndim == b.ndim == 2:
            if b.shape[-1] < a.shape[-1]:
                b = F.pad(b, (0, a.shape[-1] - b.shape[-1]))
            elif b.shape[-1] > a.shape[-1]:
                a = F.pad(a, (0, b.shape[-1] - a.shape[-1]))
        if b.shape != a.shape or b.ndim not in [1, 2]:
            raise ValueError(
                "b and a must have the same one- or two-dimensional shape (n_taps,) or "
                f"(n_filters, n_taps), got shapes {b.shape} and {a.shape}"
            )
        self.register_buffer("b", b)
        self.register_buffer("a", a)

    def forward(self, x, axis=-1, batching=False):
        """Filter input signal.

        Parameters
        ----------
        x : torch.Tensor
            Input signal.
        axis : int, optional
            Axis along which to filter.
        batching : bool, optional
            If ``True``, the input is assumed to have shape ``(..., n_filters, time)``
            and each channel is filtered with its own filter.

        Returns
        -------
        torch.Tensor
            Filtered signal.

        """
        axis = axis % x.ndim
        if batching:
            _batching_check(x, axis, self.b)
        if self.a.dtype.is_complex:
            # TODO: implement complex lfilter in torch source code
            raise ValueError("complex IIR filters are not supported")
        else:
            y = torchaudio.functional.lfilter(
                x.moveaxis(axis, -1),
                self.a.view(-1, self.a.shape[-1]),
                self.b.view(-1, self.b.shape[-1]),
                batching=batching,
                clamp=False,
            )
        if self.b.ndim == 1:
            y = y.squeeze(-2).moveaxis(-1, axis)
        else:
            y = y.moveaxis(-1, axis).moveaxis(-1, axis)
        return y


@FilterbankRegistry.register("gammatone")
class GammatoneFilterbank(nn.Module):
    """Gammatone filterbank.

    Parameters
    ----------
    fs : float, optional
        Sampling frequency.
    f_min : float, optional
        Minimum frequency. Ignored if ``fc`` is not ``None``.
    f_max : float, optional
        Maximum frequency. Ignored if ``fc`` is not ``None``.
    n_filters : int, optional
        Number of filters. Ignored if ``fc`` is not ``None``.
    fc : float or list or numpy.ndarray
        Center frequencies. If ``None``, a sequence of evenly-spaced frequencies on a
        ERB-number scale is calculated using ``f_min``, ``f_max`` and ``n_filters``.
    order : int, optional
        Filter order.
    bw_mult : float or numpy.ndarray, optional
        Bandwidth scaling factor. See :func:`gammatone` for details.
    filter_type : str, optional
        Filter type. See :func:`gammatone` for details.
    fir_ntaps : int, optional
        Number of taps for the FIR filter. Ignored if ``filter_type != "fir"``.
    fir_dur : float, optional
        Duration of FIR in seconds. Ignored if ``filter_type != "fir"``. Specify either
        ``fir_ntaps`` or ``fir_dur``, not both.
    iir_output : {"ba", "sos"}, optional
        Output format for IIR filters. See :func:`gammatone` for details.
    gain : float or list or numpy.ndarray, optional
        Gain. If :class:`numpy.ndarray`, must have shape ``(n_filters,)``. If ``None``,
        no gain is applied.
    precision : {"single", "double"}, optional
        Filter coefficient precision. If ``"double"``, the filter coefficients are
        less likely to be unstable, especially if ``iir_output == "ba"``.
    compensate_delay : bool, optional
        Whether to compensate for the delay introduced by the filters. The phase delay
        compensation is only supported by the FIR filter type and is applied directly
        to the filter coefficients (it thus has an effect during analysis). The group
        delay compensation is supported for all filter types and is applied during
        synthesis.

    """

    def __init__(
        self,
        fs=20000,
        fc=None,
        f_min=80,
        f_max=8000,
        n_filters=30,
        order=4,
        bw_mult=None,
        filter_type="apgf",
        fir_ntaps=512,
        fir_dur=None,
        iir_output="sos",
        gain=None,
        precision="single",
        compensate_delay=False,
    ):
        super().__init__()
        if fc is None:
            fc = erbspace(f_min, f_max, n_filters)
        coeffs = gammatone(
            fc,
            fs,
            order=order,
            bw_mult=bw_mult,
            filter_type=filter_type,
            fir_ntaps=fir_ntaps,
            fir_dur=fir_dur,
            iir_output=iir_output,
            compensate_delay=compensate_delay,
        )

        if precision == "single":
            dtype = torch.float32 if np.isrealobj(coeffs[0]) else torch.complex64
        elif precision == "double":
            dtype = torch.float64 if np.isrealobj(coeffs[0]) else torch.complex128
        else:
            raise ValueError(f"precision must be single or double, got {precision}")

        if filter_type == "fir":
            fbs = [FIRFilterbank(coeffs, dtype=dtype)]
        elif iir_output == "ba":
            fbs = [IIRFilterbank(*coeffs, dtype=dtype)]
        else:
            coeffs = coeffs.reshape(-1, coeffs.shape[-2], 6)
            fbs = [
                IIRFilterbank(coeffs[:, i, :3], coeffs[:, i, 3:], dtype=dtype)
                for i in range(coeffs.shape[1])
            ]
        self._fbs = nn.ModuleList(fbs)
        self.fs = fs
        self.order = order
        self.bw_mult = bw_mult
        self.compensate_delay = compensate_delay

        self.register_buffer("fc", torch.tensor(fc, dtype=dtype))

        if gain is not None:
            gain, _ = _check_0d_or_1d(gain, "gain")
            if len(gain) > 1 and len(gain) != len(fc):
                raise ValueError(
                    "gain must have length equal to number of filters, "
                    f"got {len(gain)} and {len(fc)}"
                )
            gain = 10 ** (gain / 20)
            gain = torch.tensor(gain, dtype=dtype)
            self.register_buffer("gain", gain)
        else:
            self.gain = None

    def forward(self, x, axis=-1, batching=False, ohc_loss=None):
        """Apply gammatone filterbank.

        Parameters
        ----------
        x : torch.Tensor
            Input signal. Shape ``(batch_size, ..., time)``.
        axis : int, optional
            Axis along which to filter.
        batching : bool, optional
            If ``True``, the input is assumed to have shape ``(batch_size, ...,
            n_filters, time)`` and each channel is filtered with its own filter.
        ohc_loss : torch.Tensor, optional
            Outer hair cell loss in [0, 1]. Shape ``(batch_size, n_filters)``.

        Returns
        -------
        torch.Tensor
            Filtered signal. Shape ``(batch_size, ..., n_filters, time)``.

        """
        if ohc_loss is not None:
            warnings.warn(f"ohc_loss not supported by {self.__class__.__name__}")
        for i, fb in enumerate(self._fbs):
            x = fb(x, axis=axis, batching=batching or i != 0)
        if self.gain is not None:
            x = self.gain[:, None] * x
        return x

    def inverse(self, x, _return_delayed=False):
        """Inverse filterbank.

        If ``compensate_delay`` is ``True``, the input filtered signals are delayed
        to align the envelope peaks before summation.

        Parameters
        ----------
        x : torch.Tensor
            Filtered signal.
        _return_delayed : bool, optional
            Whether to return the delayed signals before summation.

        Returns
        -------
        inv : torch.Tensor
            Reconstructed signal.
        _delayed : torch.Tensor
            Delayed signals before summation. Returned only if ``_return_delayed`` is
            ``True``.

        """
        if self.compensate_delay:
            if self.bw_mult is None:
                bw_mult = _bw_mult_from_order(self.order)
            else:
                bw_mult = self.bw_mult
            bw = 2 * np.pi * bw_mult * aud_filt_bw(self.fc)
            envelope_peak = (self.order - 1) / bw
            padding = (envelope_peak.max() - envelope_peak) * self.fs
            padding = padding.round().astype(int)
            x = torch.stack(
                [
                    F.pad(x[..., i, : x.shape[-1] - pad], (pad, 0))
                    for i, pad in enumerate(padding)
                ]
            )
            _delayed = x.moveaxis(0, -2)
        else:
            _delayed = x
        inv = _delayed.sum(dim=-2)
        return (inv, _delayed) if _return_delayed else inv


@FilterbankRegistry.register("drnl")
class DRNLFilterbank(nn.Module):
    """Dual-Resonance Non-Linear (DRNL) filterbank.

    Proposed in [1]. Same as ``lopezpoveda2001.m`` in the AMT.

    .. [1] E. Lopez-Poveda and R. Meddis, "A human nonlinear cochlear filterbank", in J.
       Acoust. Soc. Am., 2001.
    """

    def __init__(
        self,
        fs=20000,
        fc=None,
        f_min=80,
        f_max=8000,
        erb_space=1,
        n_filters=None,
        lin_ngt=2,
        lin_nlp=4,
        lin_fc=[-0.06762, 1.01679],
        lin_bw=[0.03728, 0.78563],
        lin_gain=[4.20405, -0.47909],
        lin_lp_cutoff=[-0.06762, 1.01679],
        nlin_ngt_before=3,
        nlin_ngt_after=None,
        nlin_nlp=3,
        nlin_fc_before=[-0.05252, 1.01650],
        nlin_fc_after=None,
        nlin_bw_before=[-0.03193, 0.77426],
        nlin_bw_after=None,
        nlin_lp_cutoff=[-0.05252, 1.01650],
        nlin_a=[1.40298, 0.81916],
        nlin_b=[1.61912, -0.81867],
        nlin_c=[np.log10(0.25), 0],
        nlin_d=1,
        middle_ear=True,
        filter_type="amt_classic",
        fir_ntaps=512,
        fir_dur=None,
        iir_output="sos",
        precision="single",
    ):
        super().__init__()

        dtype = torch.float32 if precision == "single" else torch.float64

        if nlin_ngt_after is None:
            nlin_ngt_after = nlin_ngt_before
        if nlin_fc_after is None:
            nlin_fc_after = nlin_fc_before
        if nlin_bw_after is None:
            nlin_bw_after = nlin_bw_before

        if fc is None:
            msg = f"Specified {erb_space=} and {n_filters=} (use only one)"
            assert np.logical_xor(erb_space is None, n_filters is None), msg
            if erb_space is not None:
                fc = erbspace_bw(f_min, f_max, erb_space)
            else:
                fc = erbspace(f_min, f_max, n_filters)

        def polfun(x, par):
            return 10 ** par[0] * x ** par[1]

        lin_fc = polfun(fc, lin_fc)
        lin_bw = polfun(fc, lin_bw)
        lin_lp_cutoff = polfun(fc, lin_lp_cutoff)
        lin_gain = polfun(fc, lin_gain)
        nlin_fc_before = polfun(fc, nlin_fc_before)
        nlin_fc_after = polfun(fc, nlin_fc_after)
        nlin_bw_before = polfun(fc, nlin_bw_before)
        nlin_bw_after = polfun(fc, nlin_bw_after)
        nlin_lp_cutoff = polfun(fc, nlin_lp_cutoff)
        nlin_a = polfun(fc, nlin_a)
        nlin_b = polfun(fc, nlin_b)
        nlin_c = polfun(fc, nlin_c)

        self.gtf_lin = GammatoneFilterbank(
            fs=fs,
            fc=lin_fc,
            order=lin_ngt,
            bw_mult=lin_bw / aud_filt_bw(lin_fc),
            filter_type=filter_type,
            fir_ntaps=fir_ntaps,
            fir_dur=fir_dur,
            iir_output=iir_output,
            precision=precision,
        )
        self.gtf_nlin_before = GammatoneFilterbank(
            fs=fs,
            fc=nlin_fc_before,
            order=nlin_ngt_before,
            bw_mult=nlin_bw_before / aud_filt_bw(nlin_fc_before),
            filter_type=filter_type,
            fir_ntaps=fir_ntaps,
            fir_dur=fir_dur,
            iir_output=iir_output,
            precision=precision,
        )
        self.gtf_nlin_after = GammatoneFilterbank(
            fs=fs,
            fc=nlin_fc_after,
            order=nlin_ngt_after,
            bw_mult=nlin_bw_after / aud_filt_bw(nlin_fc_after),
            filter_type=filter_type,
            fir_ntaps=fir_ntaps,
            fir_dur=fir_dur,
            iir_output=iir_output,
            precision=precision,
        )
        if filter_type == "fir":
            if (fir_ntaps is not None) and (fir_dur is not None):
                msg = f"{fir_ntaps=} conflicts with {fir_dur=} and {fs=}"
                assert fir_ntaps == int(fir_dur * fs), msg
            if fir_ntaps is None:
                fir_ntaps = int(fir_dur * fs)
            dirac = impulse(fir_ntaps, dtype=dtype)
            # FIR of linear path lowpass filter (applied ``lin_nlp`` times)
            lin_lp_fir = dirac.clone()
            _lin_lpf_iir = IIRFilterbank(
                *butter(2, lin_lp_cutoff / (fs / 2)),
                dtype=dtype,
            )
            for itr in range(lin_nlp):
                lin_lp_fir = _lin_lpf_iir(lin_lp_fir, batching=itr > 0)
            lin_nlp = 1
            self.lpf_lin = FIRFilterbank(lin_lp_fir)
            # FIR of nonlinear path lowpass filter (applied ``nlin_nlp`` times)
            nlin_lp_fir = dirac.clone()
            _nlin_lpf_iir = IIRFilterbank(
                *butter(2, nlin_lp_cutoff / (fs / 2)),
                dtype=dtype,
            )
            for itr in range(nlin_nlp):
                nlin_lp_fir = _nlin_lpf_iir(nlin_lp_fir, batching=itr > 0)
            nlin_nlp = 1
            self.lpf_nlin = FIRFilterbank(nlin_lp_fir)
        else:
            self.lpf_lin = IIRFilterbank(
                *butter(2, lin_lp_cutoff / (fs / 2)),
                dtype=dtype,
            )
            self.lpf_nlin = IIRFilterbank(
                *butter(2, nlin_lp_cutoff / (fs / 2)),
                dtype=dtype,
            )

        self.register_buffer("fc", torch.tensor(fc, dtype=dtype))
        self.register_buffer("lin_gain", torch.tensor(lin_gain, dtype=dtype))
        self.register_buffer("nlin_a", torch.tensor(nlin_a, dtype=dtype))
        self.register_buffer("nlin_b", torch.tensor(nlin_b, dtype=dtype))
        self.register_buffer("nlin_c", torch.tensor(nlin_c, dtype=dtype))
        self.register_buffer("nlin_d", torch.tensor(nlin_d, dtype=dtype))
        self.register_buffer("lin_nlp", torch.tensor(lin_nlp))
        self.register_buffer("nlin_nlp", torch.tensor(nlin_nlp))

        if middle_ear:
            self.middle_ear_filter = FIRFilterbank(
                middle_ear_filter(fs),
                dtype=dtype,
            )
        else:
            self.middle_ear_filter = None

    def forward(self, x, ohc_loss=None):
        """Apply DRNL filterbank.

        Parameters
        ----------
        x : torch.Tensor
            Input signal. Shape ``(batch_size, ..., time)``.
        ohc_loss : torch.Tensor, optional
            Outer hair cell loss in [0, 1]. Shape ``(batch_size, n_filters)``.

        Returns
        -------
        torch.Tensor
            Filtered signal. Shape ``(batch_size, ..., n_filters, time)``.

        """
        if ohc_loss is not None:
            if x.ndim == 1 and ohc_loss.ndim != 1:
                raise ValueError(
                    "ohc_loss must be 1D for 1D input",
                    f"got {ohc_loss.shape} ohc_loss and {x.shape} input",
                )
            if x.ndim > 1 and (ohc_loss.ndim != 2 or ohc_loss.shape[0] != x.shape[0]):
                raise ValueError(
                    "ohc_loss must be 2D with same batch size as input for batched "
                    f"inputs, got {ohc_loss.shape} ohc_loss and {x.shape} input"
                )

        unbatched = x.ndim == 1
        if unbatched:
            x = x[None, :]
            if ohc_loss is not None:
                ohc_loss = ohc_loss[None, :]

        x = x * 10 ** ((93.98 - 100) / 20)

        if self.middle_ear_filter is not None:
            x = self.middle_ear_filter(x)

        y_lin = torch.einsum("...i,...j->...ij", (self.lin_gain[None, :], x))
        y_lin = self.gtf_lin(y_lin, batching=True)
        for _ in range(self.lin_nlp):
            y_lin = self.lpf_lin(y_lin, batching=True)

        if ohc_loss is None:
            ohc_loss = 1
        else:
            ohc_loss = ohc_loss.unsqueeze(-1)  # (batch_size, n_filters, 1)
            while ohc_loss.ndim < y_lin.ndim:
                ohc_loss = ohc_loss.unsqueeze(1)  # (batch_size, ..., n_filters, 1)

        y_nlin = self.gtf_nlin_before(x, batching=False)
        tiny = torch.finfo(y_nlin.dtype).tiny  # clamping with this fixes nan gradients
        y_nlin = y_nlin.sign() * torch.minimum(
            self.nlin_a[:, None] * y_nlin.abs() ** self.nlin_d * ohc_loss,
            self.nlin_b[:, None] * y_nlin.abs().clamp(tiny) ** self.nlin_c[:, None],
        )
        y_nlin = self.gtf_nlin_after(y_nlin, batching=True)
        for _ in range(self.nlin_nlp):
            y_nlin = self.lpf_nlin(y_nlin, batching=True)

        y = y_lin + y_nlin
        if unbatched:
            y = y[0, :]
        return y


@FilterbankRegistry.register("none")
class NoFilterbank(nn.Module):
    """Identity filterbank.

    This filterbank does not perform any filtering and is useful for disabling the
    filtering step in a pipeline.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, ohc_loss=None):
        """Return input signal unchanged."""
        if ohc_loss is not None:
            raise ValueError(f"ohc_loss not supported by {self.__class__.__name__}")
        return x


def data_goode1994():
    """Get data from Goode et al. (1994).

    Same as ``data_goode1994.m`` in the AMT.
    """
    return np.array(
        [
            [400, 0.19953],
            [600, 0.22909],
            [800, 0.21878],
            [1000, 0.15136],
            [1200, 0.10000],
            [1400, 0.07943],
            [1600, 0.05754],
            [1800, 0.04365],
            [2000, 0.03311],
            [2200, 0.02754],
            [2400, 0.02188],
            [2600, 0.01820],
            [2800, 0.01445],
            [3000, 0.01259],
            [3500, 0.00900],
            [4000, 0.00700],
            [4500, 0.00457],
            [5000, 0.00500],
            [5500, 0.00400],
            [6000, 0.00300],
            [6500, 0.00275],
        ]
    )


def data_lopezpoveda2001():
    """Get data from Lopez-Poveda & Meddis (2001).

    Same as ``data_lopezpoveda2001.m`` with argument ``fig2b`` in the AMT.
    """
    data = data_goode1994()
    data[:, 1] *= 1e-6 * 2 * np.pi * data[:, 0]
    data[:, 1] *= 10 ** (-104 / 20)
    data[:, 1] *= 2**0.5
    extrp = np.array(
        [
            [100, 1.181e-9],
            [200, 2.363e-9],
            [7000, 8.705e-10],
            [7500, 8.000e-10],
            [8000, 7.577e-10],
            [8500, 7.168e-10],
            [9000, 6.781e-10],
            [9500, 6.240e-10],
            [10000, 6.000e-10],
        ]
    )
    return np.vstack(
        [
            extrp[extrp[:, 0] < data[0, 0]],
            data,
            extrp[extrp[:, 0] > data[-1, 0]],
        ]
    )


def _check_1d_or_2d(x, name="input"):
    if not isinstance(x, (list, np.ndarray)):
        raise TypeError(
            f"{name} must be list or np.ndarray, got {x.__class__.__name__}"
        )
    if isinstance(x, list):
        x = np.array(x)
    is_1d = x.ndim == 1
    if is_1d:
        x = x[None, :]
    elif x.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional, got shape {x.shape}")
    return x, is_1d


def _check_0d_or_1d(x, name="input"):
    is_0d = (
        isinstance(x, (int, float, np.integer, np.floating))
        or isinstance(x, np.ndarray)
        and x.ndim == 0
    )
    if is_0d:
        x = np.array([x])
    elif isinstance(x, list):
        x = np.array(x)
    elif not isinstance(x, np.ndarray):
        raise TypeError(
            f"{name} must be scalar or np.ndarray, got {x.__class__.__name__}"
        )
    if x.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {x.shape}")
    return x, is_0d


def _batching_check(x, axis, b):
    if axis != x.ndim - 1:
        raise ValueError("batching only supported when filtering along last axis")
    if x.ndim < 2:
        raise ValueError("batching requires input with at least two dimensions")
    if b.ndim != 2:
        raise ValueError("batching requires filter to be two-dimensional")
    if x.shape[-2] != b.shape[-2]:
        raise ValueError(
            "batching requires input and filter to have the same number of "
            f"channels, got {x.shape[-2]} and {b.shape[-2]}"
        )


def _bw_mult_from_order(order):
    return math.factorial(order - 1) ** 2 / (
        np.pi * math.factorial(2 * order - 2) * 2 ** (-2 * order + 2)
    )
