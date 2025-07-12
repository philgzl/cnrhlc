import torch

from cnrhlc.bsrnn import BSRNN


def test_forward():
    net = BSRNN()
    x = torch.randn(1, 1, 257, 100, dtype=torch.cfloat)
    output = net(x)
    assert output.shape == x.shape


def test_forward_with_embedding():
    net = BSRNN(emb_dim=20)
    x = torch.randn(1, 1, 257, 100, dtype=torch.cfloat)
    emb = torch.randn(1, 20)
    output = net(x, emb)
    assert output.shape == x.shape
