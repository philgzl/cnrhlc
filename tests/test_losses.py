import pytest
import torch
import torch.nn.functional as F

from cnrhlc.losses import LossRegistry

BATCH_SIZE = 8
CHANNELS = 2
MIN_LENGTH = 4000
MAX_LENGTH = 16000


@pytest.mark.parametrize("loss", LossRegistry.keys())
def test_loss(torch_rng, loss):
    # randomize lengths
    lengths = torch.randint(MIN_LENGTH, MAX_LENGTH, (BATCH_SIZE,), generator=torch_rng)

    # create inputs with length lengths
    inputs = [torch.randn(CHANNELS, length, generator=torch_rng) for length in lengths]

    # pad inputs and make batch
    batched_inputs = torch.stack([F.pad(x, (0, MAX_LENGTH - x.shape[-1])) for x in inputs])

    # mimic neural net processing
    def dummy_net_func(x):
        return x + torch.randn(*x.shape, generator=torch_rng)

    batched_outputs = dummy_net_func(batched_inputs)
    outputs = [x[..., :length] for x, length in zip(batched_outputs, lengths)]

    # create targets
    targets = [torch.randn(CHANNELS, length, generator=torch_rng) for length in lengths]

    # pad targets and make batch
    batched_targets = torch.stack([F.pad(x, (0, MAX_LENGTH - x.shape[-1])) for x in targets])

    # init loss
    loss_cls = LossRegistry.get(loss)
    loss_obj = loss_cls()
    loss_obj.eval()

    with torch.no_grad():
        # 2 ways of calculating: either batch processing...
        batched_losses = loss_obj(batched_outputs, batched_targets, lengths)

        # ...or one-by-one
        losses = torch.tensor(
            [
                loss_obj(x.unsqueeze(0), y.unsqueeze(0), torch.tensor([length]))
                for x, y, length in zip(outputs, targets, lengths)
            ]
        )

    # both should give the same result
    assert torch.allclose(batched_losses, losses)
