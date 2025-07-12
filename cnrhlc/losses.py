from abc import abstractmethod
from typing import override

import torch
import torch.nn as nn

from .auditory import AuditoryModel
from .utils import Registry, apply_mask

LossRegistry = Registry("loss")
"""Registry for loss functions.

Members of this registry are:

.. list-table::
    :header-rows: 1

    * - Key
      - Member
    * - ``"pnorm"``
      - :class:`PNormLoss`
    * - ``"l1"``
      - :class:`L1Loss`
    * - ``"l2"``
      - :class:`L2Loss`
    * - ``"mse"``
      - :class:`MSELoss`
    * - ``"auditory"``
      - :class:`AuditoryLoss`
    * - ``"cnrhlc"``
      - :class:`ControllableNoiseReductionHearingLossCompensationLoss`

"""


class BaseLoss(nn.Module):
    """Base class for all losses.

    Losses are always computed along the last dimension of the input tensors. If there
    are remaining dimensions other that the batch dimension after reducing the last
    dimension, then these are reduced by taking the mean.

    Subclasses must implement the :meth:`compute` method.

    """

    def forward(self, x, y, lengths=None, weight=None, audiogram=None):
        """Compute the loss.

        Performs checks before calling the :meth:`compute` method.

        Parameters
        ----------
        x : torch.Tensor
            Predictions. Shape ``(batch_size, ..., time)``.
        y : torch.Tensor or float
            Targets. Can be a float or a tensor with shape ``(batch_size, ..., time)``.
        lengths : torch.Tensor, optional
            Length of tensors along last axis before batching. Shape ``(batch_size,)``.
        weight : torch.Tensor, optional
            Weight for each tensor. Shape ``(batch_size,)``.
        audiogram : torch.Tensor, optional
            Audiogram.

        Returns
        -------
        torch.Tensor
            Loss. Shape ``(batch_size,)``.

        """
        if lengths is None:
            lengths = torch.full((x.shape[0],), x.shape[-1], device=x.device)
        if isinstance(y, float):
            y = torch.full_like(x, y)
        assert x.shape == y.shape
        assert x.ndim >= 2
        return self.compute(x, y, lengths, weight=weight, audiogram=audiogram)

    @abstractmethod
    def compute(self, x, y, lengths, weight=None, audiogram=None):
        """Compute the loss.

        This method should not be called directly. Use :meth:`forward` instead.

        Parameters
        ----------
        x : torch.Tensor
            Predictions. Shape ``(batch_size, ..., length)``.
        y : torch.Tensor
            Targets. Shape ``(batch_size, ..., length)``.
        lengths : torch.Tensor
            Length of tensors along last axis before batching. Shape ``(batch_size,)``.
        weight : torch.Tensor, optional
            Weight for each tensor. Shape ``(batch_size,)``.
        audiogram : torch.Tensor, optional
            Audiogram.

        Returns
        -------
        torch.Tensor
            Loss. Shape ``(batch_size,)``.

        """
        raise NotImplementedError


@LossRegistry.register("pnorm")
class PNormLoss(BaseLoss):
    r"""P-norm loss.

    Calculated as

    .. math::

        \frac{1}{N} \left( \sum_{i=1}^{N} |x_i - y_i|^p \right)^{1/p}.

    Parameters
    ----------
    order : int, optional
        Order of the norm :math:`p`.
    norm : bool, optional
        If ``False``, then the :math:`1/p` exponent is not applied.

    """

    def __init__(self, order=2, norm=True):
        super().__init__()
        self.order = order
        self.norm = norm

    @override
    def compute(self, x, y, lengths, weight=None, audiogram=None):
        assert (
            audiogram is None
        ), f"audiogram argument not supported for {self.__class__.__name__} loss"
        x, y = apply_mask(x, y, lengths=lengths)
        loss = (x - y).abs().pow(self.order).sum(-1)
        if self.norm:
            loss = loss.pow(1 / self.order)
        loss /= lengths.view(-1, *[1] * (x.ndim - 2))
        if weight is not None:
            loss *= weight.view(-1, *[1] * (x.ndim - 2))
        dims = tuple(range(1, x.ndim - 1))
        if dims:
            loss = loss.mean(dims)
        return loss


@LossRegistry.register("l1")
class L1Loss(PNormLoss):
    """L1 loss."""

    def __init__(self):
        super().__init__(order=1)


@LossRegistry.register("l2")
class L2Loss(PNormLoss):
    """L2 loss."""

    def __init__(self):
        super().__init__(order=2)


@LossRegistry.register("mse")
class MSELoss(PNormLoss):
    """Mean squared error (MSE) loss."""

    def __init__(self):
        super().__init__(order=2, norm=False)


@LossRegistry.register("auditory")
class AuditoryLoss(BaseLoss):
    """Auditory model-based loss.

    Parameters
    ----------
    am_kw : dict, optional
        Keyword arguments for the auditory model. See
        :class:`~.auditory.AuditoryModel`
    am_kw_hi : dict, optional
        Keyword arguments for the hearing impaired auditory model. If ``None``, defaults
        to ``am_kw``.
    am_kw_nh : dict, optional
        Keyword arguments for the normal hearing auditory model. If ``None``, defaults
        to ``am_kw``.
    loss : str, optional
        Loss function between the auditory model outputs. Must be a member of
        :data:`LossRegistry`.
    loss_kw : dict, optional
        Keyword arguments for the loss function.

    """

    def __init__(
        self,
        am_kw=None,
        am_kw_hi=None,
        am_kw_nh=None,
        loss="mse",
        loss_kw=None,
    ):
        super().__init__()
        self.am_hi = AuditoryModel(**(am_kw_hi or am_kw or {}))
        self.am_nh = AuditoryModel(**(am_kw_nh or am_kw or {}))
        self.loss = LossRegistry.get(loss)(**(loss_kw or {}))

    @override
    def compute(self, x, y, lengths, weight=None, audiogram=None):
        x = self.am_hi(x, audiogram=audiogram)
        y = self.am_nh(y, audiogram=None)
        return self.loss(x, y, lengths, weight=weight, audiogram=None)


@LossRegistry.register("cnrhlc")
class ControllableNoiseReductionHearingLossCompensationLoss(BaseLoss):
    """Controllable noise reduction and hearing loss compensation loss.

    Parameters
    ----------
    am_kw : dict, optional
        Keyword arguments for the auditory model. See
        :class:`~.auditory.AuditoryModel`
    am_kw_hi : dict, optional
        Keyword arguments for the hearing impaired auditory model. If ``None``, defaults
        to ``am_kw``.
    am_kw_nh : dict, optional
        Keyword arguments for the normal hearing auditory model. If ``None``, defaults
        to ``am_kw``.
    loss : str, optional
        Loss function between the auditory model outputs. Must be a member of
        :data:`LossRegistry`.
    loss_kw : dict, optional
        Keyword arguments for the loss function.
    nh_denoising : bool, optional
        If ``True``, the noise reduction loss term is computed from the normal hearing
        representation of the clean and processed signals. Else, it is computed from
        their hearing impaired representation.

    """

    def __init__(
        self,
        am_kw=None,
        am_kw_hi=None,
        am_kw_nh=None,
        loss="mse",
        loss_kw=None,
        nh_denoising=True,
    ):
        super().__init__()
        self.am_hi = AuditoryModel(**(am_kw_hi or am_kw or {}))
        self.am_nh = AuditoryModel(**(am_kw_nh or am_kw or {}))
        self.loss = LossRegistry.get(loss)(**(loss_kw or {}))
        self.nh_denoising = nh_denoising
        self.log_uncertainty_denoising = nn.Parameter(torch.zeros(1))
        self.log_uncertainty_compensation = nn.Parameter(torch.zeros(1))

    @override
    def compute(self, x, y, lengths, weight=None, audiogram=None):
        assert x.ndim == y.ndim == 3  # (batch_size, n_channels, time)
        compensated, denoised = x.unbind(1)
        noisy, clean = y.unbind(1)
        if self.nh_denoising:
            denoising_loss = self.loss(
                self.am_nh(denoised, audiogram=None),
                self.am_nh(clean, audiogram=None),
                lengths,
                weight=weight,
                audiogram=None,
            )
        else:
            denoising_loss = self.loss(
                self.am_hi(denoised, audiogram=audiogram),
                self.am_hi(clean, audiogram=audiogram),
                lengths,
                weight=weight,
                audiogram=None,
            )
        compensation_loss = self.loss(
            self.am_hi(compensated, audiogram=audiogram),
            self.am_nh(noisy, audiogram=None),
            lengths,
            weight=weight,
            audiogram=None,
        )
        return (
            torch.exp(-self.log_uncertainty_denoising) * denoising_loss
            + torch.exp(-self.log_uncertainty_compensation) * compensation_loss
            + self.log_uncertainty_denoising
            + self.log_uncertainty_compensation
        )
