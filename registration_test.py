import torch
import pytest

# =======================================================
# torch.linalg.cholesky_ex and torch.linalg.cholesky
# run same kernel under the hood
# linalg_cholesky_ex_kernel_impl

@pytest.mark.parametrize("upper", [True, False])
def test_linalg_cholesky_ex(upper):
    A = torch.tensor([[4.0, 2.0], [2.0, 3.0]]).to("xpu")
    L, info = torch.linalg.cholesky_ex(A, upper=upper)

@pytest.mark.parametrize("upper", [True, False])
def test_linalg_cholesky_ex_out(upper):
    A = torch.tensor([[4.0, 2.0], [2.0, 3.0]]).to("xpu")
    L_out = torch.empty(2, 2, device="xpu")
    info_out = torch.empty((), dtype=torch.int32, device="xpu")
    L, info = torch.linalg.cholesky_ex(A, upper=upper, out=(L_out, info_out))

@pytest.mark.parametrize("upper", [True, False])
def test_linalg_cholesky(upper):
    A = torch.tensor([[4.0, 2.0], [2.0, 3.0]]).to("xpu")
    L = torch.linalg.cholesky(A, upper=upper)

@pytest.mark.parametrize("upper", [True, False])
def test_linalg_cholesky_out(upper):
    A = torch.tensor([[4.0, 2.0], [2.0, 3.0]]).to("xpu")
    L_out = torch.empty(2, 2, device="xpu")
    L = torch.linalg.cholesky(A, upper=upper, out=L_out)


# =======================================================
# torch.cholesky (deprecated one)
# has separate kernels for cholesky.out and cholesky in upstream
# we can either implement it as single kernel or not implement at all

@pytest.mark.parametrize("upper", [True, False])
def test_deprecated_cholesky(upper):
    A = torch.tensor([[4.0, 2.0], [2.0, 3.0]]).to("xpu")
    L = torch.cholesky(A, upper=upper)
    # NotImplementedError: The operator 'aten::cholesky' is not currently implemented for the XPU device.

@pytest.mark.parametrize("upper", [True, False])
def test_deprecated_cholesky_out(upper):
    A = torch.tensor([[4.0, 2.0], [2.0, 3.0]]).to("xpu")
    L_out = torch.empty(2, 2, device="xpu")
    L = torch.cholesky(A, upper=upper, out=L_out)
    # NotImplementedError: The operator 'aten::cholesky.out' is not currently implemented for the XPU device.
