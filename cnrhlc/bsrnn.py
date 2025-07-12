import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


class BSRNN(nn.Module):
    """Band-Split RNN (BSRNN). Proposed in [1], [2] and [3].

    This implementations includes the residual spectrogram from [3] in addition to the
    multiplicative mask.

    This implementation also has different connections between the layers compared to
    the papers. Here all the layer outputs are connected à la Conv-TasNet, while in the
    papers the connections are between the input and the output of each layer only.

    Parameters
    ----------
    input_channels : int, optional
        Number of input microphones.
    reference_channels : list, optional
        Reference channels to apply predicted masks to. The number of output channels is
        the length of the list.
    n_fft : int, optional
        Number of FFT points used in the STFT.
    fs : int, optional
        Sampling frequency.
    base_channels : int, optional
        Number of base channels.
    layers : int, optional
        Number of layers.
    subband_right_limits : list, optional
        Right limit of each subband in Hz. If not provided, the subbands are defined
        as in [3].
    emb_dim : int, optional
        Dimension of the external embedding vector.
    real_mask : bool, optional
        If ``True``, the output multiplicative mask is real-valued.
    residual_spectrogram : bool, optional
        If ``False``, no residual spectrogram is predicted.

    .. [1] Y. Luo and J. Yu, "Music source separation with band-split RNN", in
       IEEE/ACM Trans. Audio, Speech, Lang. Process., 2023.
    .. [2] J. Yu and Y. Luo, "Efficient monaural speech enhancement with universal
       sample rate band-split RNN", in Proc. ICASSP, 2023.
    .. [3] J. Yu, H. Chen, Y. Luo, R. Gu and C. Weng, "High fidelity speech
       enhancement with band-split RNN", in Proc. Interspeech, 2023.

    """

    def __init__(
        self,
        input_channels=1,
        reference_channels=[0],
        n_fft=512,
        fs=16000,
        base_channels=64,
        layers=6,
        subband_right_limits=None,
        emb_dim=None,
        real_mask=False,
        residual_spectrogram=True,
    ):
        super().__init__()
        if subband_right_limits is None:
            # The bands are defined differently in [2] and [3]. We follow [3], namely
            # "we split the spectrogram into 33 subbands, including twenty 200 Hz
            # bandwidth subbands for low frequency followed by six 500 Hz subbands and
            # seven 2 kHz subbands".
            bandwidths = [200] * 20 + [500] * 6 + [2000] * 7
            subband_right_limits = list(itertools.accumulate(bandwidths))
        subbands = self._build_subbands(n_fft, fs, subband_right_limits)
        self.band_split = _BandSplit(subbands, input_channels, base_channels)
        self.time_rnns = nn.ModuleList(
            [_RNNBlock(base_channels, 2 * base_channels) for i in range(layers)]
        )
        self.freq_rnns = nn.ModuleList(
            [_RNNBlock(base_channels, 2 * base_channels) for i in range(layers)]
        )
        if emb_dim is None:
            self.emb_layer = None
        else:
            self.emb_layer = nn.Sequential(
                nn.Linear(emb_dim, 2 * len(subbands) * base_channels),
                nn.Tanh(),
            )
        self.separator = _Separator(
            subbands,
            base_channels,
            len(reference_channels),
            real_mask,
            residual_spectrogram,
        )
        self.reference_channels = reference_channels

    @staticmethod
    def _build_subbands(n_fft, fs, right_limits):
        # returns a list of subband indexes in the form (start, end)
        df = fs / n_fft
        rfftfreqs = torch.arange(n_fft // 2 + 1) * df
        right_limits_idx = [
            torch.where(rfftfreqs > right_limit)[0][0].item()
            for right_limit in right_limits
            if right_limit < rfftfreqs[-1]
        ]
        # add last subband
        last_right_limit_idx = min(n_fft // 2 + 1, right_limits[-1] // df + 1)
        if right_limits_idx[-1] < last_right_limit_idx:
            right_limits_idx.append(last_right_limit_idx)
        subbands = [
            (0 if i == 0 else right_limits_idx[i - 1], right_limits_idx[i])
            for i in range(len(right_limits_idx))
        ]
        return subbands

    def forward(self, input, emb=None, return_mask=False):
        """Forward pass.

        Parameters
        ----------
        input : torch.Tensor
            Input STFT. Shape ``(batch_size, num_mics, num_bins, num_frames)```.
        emb : torch.Tensor, optional
            Embedding vector. Shape ``(batch_size, emb_dim)``.
        return_mask : bool, optional
            Whether to also return the mask and the residual spectrogram.

        Returns
        -------
        enhanced : torch.Tensor
            Enhanced STFT. Shape ``(batch_size, num_outputs, num_bins, num_frames)``.
        mask : torch.Tensor
            Mask and residual spectrogram. Returned only if ``return_mask`` is ``True``.
            Shape ``(batch_size, num_outputs, num_bins, num_frames)`` if
            ``residual_spectrogram`` is ``False``, else ``(batch_size, num_outputs,
            num_bins, num_frames, 2)``, in which case the mask and the residual
            spectrogram can be recovered with ``mask, res = mask.unbind(dim=-1)``.

        """
        # input is complex with shape (B, M, F, T)
        assert not (emb is None and self.emb_layer is not None)
        assert not (emb is not None and self.emb_layer is None)
        x = torch.view_as_real(input)  # (B, M, F, T, 2)
        x = self.band_split(x)  # (B, N, K, T)
        if emb is not None:
            # FiLM
            emb = self.emb_layer(emb)  # (B, N * K * 2)
            emb = emb.reshape(emb.shape[0], x.shape[1], x.shape[2], 2)  # (B, N, K, 2)
            emb = emb.unsqueeze(-2)  # (B, N, K, 1, 2)
            scale, shift = emb[..., 0], emb[..., 1]  # (B, N, K, 1)
        skip = x
        for time_rnn, freq_rnn in zip(self.time_rnns, self.freq_rnns):
            if emb is not None:
                x = x * (1 + scale) + shift
            # rnn across time
            x = time_rnn(x)
            skip = skip + x
            # rnn across subbands
            x = x.transpose(-1, -2)  # swap time and subband dimensions
            x = freq_rnn(x)
            x = x.transpose(-1, -2)  # revert the swap
            skip = skip + x
        mask, res = self.separator(skip)  # (B, output_channels, F, T, 1 or 2)
        if mask.shape[-1] == 1:
            mask = mask.squeeze(-1)  # (B, output_channels, F, T)
        else:
            mask = torch.view_as_complex(mask)  # (B, output_channels, F, T)
        # pad mask and res to span full frequency range
        # padding can be negative if input has lower sampling rate
        mask = F.pad(mask, (0, 0, 0, input.shape[2] - mask.shape[2]))
        output = mask * input[:, self.reference_channels, :, :]
        if res is not None:
            res = torch.view_as_complex(res)  # (B, output_channels, F, T)
            res = F.pad(res, (0, 0, 0, input.shape[2] - res.shape[2]))
            output = output + res
        if return_mask:
            return output, (mask if res is None else torch.stack([mask, res], dim=-1))
        return output


class _BandSplit(nn.Module):
    def __init__(self, subbands, input_channels, channels):
        super().__init__()
        self.subbands = subbands
        self.norm = nn.ModuleList(
            [
                nn.GroupNorm(1, 2 * input_channels * (end - start))
                for start, end in subbands
            ]
        )
        self.fc = nn.ModuleList(
            [
                nn.Conv1d(2 * input_channels * (end - start), channels, 1)
                for start, end in subbands
            ]
        )

    def forward(self, input):
        B, M, _, T, _ = input.shape
        out = []
        for i, (start, end) in enumerate(self.subbands):
            x = input[:, :, start:end, :, :]
            # subband can be empty if input has lower sampling rate
            # if so then skip the subband
            if x.shape[2] == 0:
                continue
            # if subband is not empty but not full then pad it
            n_bins = end - start
            if x.shape[2] < n_bins:
                x = F.pad(x, (0, 0, 0, 0, 0, n_bins - x.shape[2]))
            x = x.moveaxis(-1, 1).reshape(B, 2 * M * n_bins, T)
            x = self.norm[i](x)
            x = self.fc[i](x)  # (B, N, T)
            out.append(x)
        return torch.stack(out, dim=2)  # (B, N, K, T)


class _RNNBlock(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        self.norm = nn.GroupNorm(1, input_channels)
        self.lstm = nn.LSTM(
            input_channels,
            hidden_channels,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(2 * hidden_channels, input_channels)

    def forward(self, x):
        B, N, K, T = x.shape
        x = self.norm(x)
        x = x.moveaxis(1, -1).reshape(-1, T, N)  # (B * K, T, N)
        x, _ = self.lstm(x)  # (B * K, T, hidden_channels)
        x = self.fc(x)  # (B * K, T, N)
        x = x.reshape(B, K, T, N).moveaxis(-1, 1)  # (B, N, K, T)
        return x


class _MLP(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.norm = nn.GroupNorm(1, input_channels)
        self.fc_1 = nn.Conv1d(input_channels, 4 * input_channels, 1)
        self.act_1 = nn.Tanh()
        self.fc_2 = nn.Conv1d(4 * input_channels, 2 * output_channels, 1)
        self.act_2 = nn.GLU(dim=1)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc_1(x)
        x = self.act_1(x)
        x = self.fc_2(x)
        x = self.act_2(x)
        return x


class _Separator(nn.Module):
    def __init__(
        self,
        subbands,
        input_channels,
        output_channels,
        real_mask,
        residual_spectrogram,
    ):
        super().__init__()
        self.output_channels = output_channels
        self.mlp_mask = nn.ModuleList(
            [
                _MLP(
                    input_channels,
                    (2 - real_mask) * output_channels * (end - start),
                )
                for start, end in subbands
            ]
        )
        self.mlp_res = (
            nn.ModuleList(
                [
                    _MLP(input_channels, 2 * output_channels * (end - start))
                    for start, end in subbands
                ]
            )
            if residual_spectrogram
            else None
        )
        self.real_mask = real_mask

    def forward(self, input):
        B, N, K, T = input.shape
        mask = []
        for i in range(K):
            x = input[:, :, i, :]
            submask = self.mlp_mask[i](x)
            submask = submask.reshape(
                B, (2 - self.real_mask), self.output_channels, -1, T
            )
            mask.append(submask)
        mask = torch.cat(mask, dim=3).moveaxis(1, -1).contiguous()
        if self.mlp_res:
            res = []
            for i in range(K):
                subres = self.mlp_res[i](x)
                subres = subres.reshape(B, 2, self.output_channels, -1, T)
                res.append(subres)
            res = torch.cat(res, dim=3).moveaxis(1, -1).contiguous()
        else:
            res = None
        return mask, res  # (B, output_channels, F, T, 1 or 2)
