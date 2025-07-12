import importlib

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--amt_path", help="path to AMT")


@pytest.fixture(scope="session")
def matlab_engine(request):
    amt_path = request.config.getoption("--amt_path")
    if amt_path is None:
        pytest.skip("AMT path is not provided")
    matlab_engine = importlib.import_module("matlab.engine")
    eng = matlab_engine.start_matlab()
    eng.beep("off")
    eng.addpath(eng.genpath(amt_path))
    eng.amt_start(nargout=0)
    return eng


@pytest.fixture(scope="function")
def torch_rng():
    g = torch.Generator()
    g.manual_seed(0)
    return g
