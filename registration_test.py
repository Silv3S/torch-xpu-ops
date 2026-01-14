import torch
import pytest

@pytest.mark.parametrize("upper", [True, False])
def test_linalg_cholesky_ex(upper):
    A = torch.tensor([[4.0, 2.0], [2.0, 3.0]]).to("xpu")
    L, info = torch.linalg.cholesky_ex(A, upper=upper)
    # Hit assertion for upper=True
