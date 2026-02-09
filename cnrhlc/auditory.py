import torch
import torch.nn as nn

from .filters import FilterbankRegistry, IIRFilterbank, butter
from .utils import Registry, linear_interpolation

IHCRegistry = Registry("rectification")
"""Registry for inner hair cell transduction stages.

Members of this registry are:

.. list-table::
    :header-rows: 1

    * - Key
      - Member
    * - ``"hwrlp"``
      - :class:`HalfwaveRectificationLowpassIHC`
    * - ``"hwr"``
      - :class:`HalfwaveRectificationIHC`
    * - ``"none"``
      - :class:`NoIHC`

"""

AdaptationRegistry = Registry("adaptation")
"""Registry for adaptation stages.

Members of this registry are:

.. list-table::
    :header-rows: 1

    * - Key
      - Member
    * - ``"log"``
      - :class:`LogAdaptation`
    * - ``"none"``
      - :class:`NoAdaptation`

"""


class AuditoryModel(nn.Module):
    """Auditory model.

    Parameters
    ----------
    fs : int, optional
        Sampling frequency.
    filterbank : str, optional
        Filterbank type. Must be a member of :data:`~.filters.FilterbankRegistry`.
    filterbank_kw : dict, optional
        Keyword arguments for the filterbank.
    ihc : str, optional
        Inner hair cell transduction stage. Must be a member of :data:`IHCRegistry`.
    ihc_kw : dict, optional
        Keyword arguments for the inner hair cell transduction stage.
    adaptation : str, optional
        Adaptation stage. Must be a member of :data:`AdaptationRegistry`.
    adaptation_kw : dict, optional
        Keyword arguments for the adaptation stage.
    modulation : str, optional
        Modulation stage. Must be a member of :data:`~.filters.FilterbankRegistry`.
    modulation_kw : dict, optional
        Keyword arguments for the modulation stage.
    output_scale : float, optional
        Output scaling factor.

    """

    def __init__(
        self,
        fs=20000,
        filterbank="gammatone",
        filterbank_kw=None,
        ihc="hwrlp",
        ihc_kw=None,
        adaptation="log",
        adaptation_kw=None,
        modulation="none",
        modulation_kw=None,
        output_scale=1.0,
    ):
        super().__init__()

        filterbank_kw = filterbank_kw or {}
        ihc_kw = ihc_kw or {}
        adaptation_kw = adaptation_kw or {}
        modulation_kw = modulation_kw or {}
        for kw, name in zip(
            [filterbank_kw, ihc_kw, adaptation_kw, modulation_kw],
            ["filterbank_kw", "ihc_kw", "adaptation_kw", "modulation_kw"],
        ):
            if "fs" not in kw:
                kw["fs"] = fs
            elif fs != kw["fs"]:
                raise ValueError(f"fs argument does not match {name}['fs']")

        self.filterbank = FilterbankRegistry.get(filterbank)(**filterbank_kw)
        self.ihc = IHCRegistry.get(ihc)(**ihc_kw)
        self.adaptation = AdaptationRegistry.get(adaptation)(**adaptation_kw)
        self.modulation = FilterbankRegistry.get(modulation)(**modulation_kw)
        self.output_scale = output_scale

    def forward(self, x, audiogram=None):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input signal. Shape ``(batch_size, ..., time)``.
        audiogram : torch.Tensor
            Audiogram. Shape ``(batch_size, n_thresholds, 2)``. First column is frequency in Hz, second column is
            hearing loss in dB.

        Returns
        -------
        torch.Tensor
            Auditory model output.

        """
        if audiogram is None:
            ohc_loss, ihc_loss = None, None
        else:
            if audiogram.shape[-1] != 2:
                raise ValueError(f"audiogram dimension along last axis must be 2, got {audiogram.shape}")
            if x.ndim == 1 and audiogram.ndim != 2:
                raise ValueError(
                    f"audiogram must be 2D for 1D input, got {audiogram.shape} audiogram and {x.shape} input"
                )
            if x.ndim > 1 and (audiogram.ndim != 3 or audiogram.shape[0] != x.shape[0]):
                raise ValueError(
                    f"audiogram must be 3D with same batch size as input for batched inputs, got {audiogram.shape} "
                    f"audiogram and {x.shape} input"
                )
            ohc_loss, ihc_loss = audiogram_to_ohc_ihc_loss(audiogram, freqs=self.filterbank.fc)
        x = self.filterbank(x, ohc_loss=ohc_loss)
        x = self.ihc(x)
        if ihc_loss is not None:
            ihc_loss = ihc_loss.unsqueeze(-1)  # (batch_size, n_filters, 1)
            while ihc_loss.ndim < x.ndim:
                ihc_loss = ihc_loss.unsqueeze(1)  # (batch_size, ..., n_filters, 1)
            x = x * ihc_loss
        x = self.adaptation(x)
        x = self.modulation(x)
        return x * self.output_scale


@IHCRegistry.register("hwrlp")
class HalfwaveRectificationLowpassIHC(nn.Module):
    """Half-wave rectification followed by low-pass filtering.

    Parameters
    ----------
    fc : float, optional
        Low-pass cutoff frequency.
    fs : float, optional
        Sampling frequency.
    order : int, optional
        Filter order.
    dtype : torch.dtype, optional
        Data type.

    """

    def __init__(
        self,
        fc=1000,
        fs=20000,
        order=2,
        dtype=torch.float32,
    ):
        super().__init__()
        b, a = butter(order, 2 * fc / fs)
        self.lp = IIRFilterbank(b, a, dtype=dtype)

    def forward(self, x):
        """Forward pass."""
        x = x.relu()
        x = self.lp(x)
        x = x.relu()
        return x


@IHCRegistry.register("hwr")
class HalfwaveRectificationIHC(nn.Module):
    """Half-wave rectification."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        """Forward pass."""
        x = x.relu()
        return x


@IHCRegistry.register("none")
class NoIHC(nn.Module):
    """Identity inner hair cell."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        """Return input unchanged."""
        return x


@AdaptationRegistry.register("log")
class LogAdaptation(nn.Module):
    r"""Instantaneous logarithmic adaptation.

    The output is calculated as :math:`\\log(1 + x / thr)`.

    """

    def __init__(self, thr=1e-5, **kwargs):
        super().__init__()
        self.thr = thr

    def forward(self, x):
        """Forward pass."""
        assert x.ge(0).all()
        return x.div(self.thr).log1p()


@AdaptationRegistry.register("none")
class NoAdaptation(nn.Module):
    """Identity adaptation."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        """Return input unchanged."""
        return x


def audiogram_to_ohc_ihc_loss(audiogram, freqs=None):
    """Compute OHC and IHC loss from audiogram.

    Parameters
    ----------
    audiogram : torch.Tensor
        Audiogram. Shape ``(batch_size, n_thresholds, 2)``. First column is frequency in Hz, second column is hearing
        loss in dB.
    freqs : torch.Tensor, optional
        Frequencies to interpolate the audiogram at. Shape ``(batch_size, n_freqs)`` or ``(n_freqs,)``. If ``None``,
        uses the input audiogram frequencies.

    Returns
    -------
    ohc_loss : torch.Tensor
        OHC loss in [0, 1]. Shape ``(batch_size, n_freqs)``.
    ihc_loss : torch.Tensor
        IHC loss in [0, 1]. Shape ``(batch_size, n_freqs)``.

    """
    max_ohc_loss = torch.tensor(
        [
            [250, 18.5918602171780],
            [375, 23.0653774100513],
            [500, 25.2602607820868],
            [750, 30.7013288310918],
            [1000, 34.0272671055467],
            [1500, 38.6752655699390],
            [2000, 39.5318838824221],
            [3000, 39.4930128714544],
            [4000, 39.3156363872299],
            [6000, 40.5210536471565],
        ],
        device=audiogram.device,
        dtype=audiogram.dtype,
    )
    if freqs is None:
        total_loss = audiogram[..., 1]
    else:
        max_ohc_loss = linear_interpolation(
            torch.log10(freqs), torch.log10(max_ohc_loss[:, 0]), max_ohc_loss[:, 1]
        ).clamp(0, 105)
        total_loss = linear_interpolation(torch.log10(freqs), torch.log10(audiogram[..., 0]), audiogram[..., 1]).clamp(
            0, 105
        )
    # 2/3 OHC loss and 1/3 IHC loss
    ohc_loss = torch.clamp(2 / 3 * total_loss, max=max_ohc_loss)
    ihc_loss = total_loss - ohc_loss
    ohc_loss = 10 ** (-ohc_loss / 20)
    ihc_loss = 10 ** (-ihc_loss / 20)
    return ohc_loss, ihc_loss
