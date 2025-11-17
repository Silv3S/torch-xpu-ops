import pytest
import torch
from torch.func import vjp, vmap, jvp


@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_vmap_mixed_batching(device):
    primal_in = torch.tensor(12.34, device=device, requires_grad=True)
    cotangent_in = torch.tensor([1.0, 1.0], device=device)
    
    def push_vjp(primal_in, cotangent_in):
        _, vjp_fn = vjp(torch.nn.functional.logsigmoid, primal_in)
        (grad,) = vjp_fn(cotangent_in)
        return grad
    
    def jvp_of_vjp(primal_in, cotangent_in, primal_tangent_in, cotangent_tangent_in):
        return jvp(
            push_vjp,
            (primal_in, cotangent_in),
            (primal_tangent_in, cotangent_tangent_in),
        )

    vmap(jvp_of_vjp, in_dims=(None, 0, None, None))(
        primal_in,
        cotangent_in,
        torch.tensor(1.0, device=device),
        torch.tensor(1.0, device=device)
    )

    # assert False
