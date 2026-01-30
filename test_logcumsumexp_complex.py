import torch
import pytest

@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_logcumsumexp_complex(dtype):
    a = torch.tensor([1.0000e-18+1.0000e+04j, 1.0000e+02+1.0000e-08j], dtype=dtype)
    x = a.to("xpu")

    print(a)
    loga = torch.logcumsumexp(a, dim=0)
    logx = torch.logcumsumexp(x, dim=0)

    print(loga)
    print(logx)
    assert torch.allclose(logx.cpu(), loga, atol=1e-5, rtol=1e-5)
