import torch
from torch.testing import make_tensor
from torch.testing._internal.common_utils import set_rng_seed

def example_input():
    set_rng_seed(2877894469)    # Need this seed for segfault reproduction
    return make_tensor((5, 5), device="xpu", dtype=torch.complex128, requires_grad=False)

# Add del s1 before calling to_sparse on t2, which seems to prevent the segfault.
def test_del_workaround():
    t1 = example_input()
    s1 = t1.to_sparse()
    t2 = example_input()
    del s1
    s2 = t2.to_sparse(1)

# Call to_sparse on t2 before calling to_sparse on t1, which seems to prevent the segfault.
def test_order_workaround():
    t1 = example_input()
    t2 = example_input()
    s1 = t1.to_sparse()
    s2 = t2.to_sparse(1)

# Remove assignment to s1/s2 variables, which seems to prevent the segfault.
def test_assignment_workaround():
    t1 = example_input()
    t1.to_sparse()
    t2 = example_input()
    t2.to_sparse(1)

# Segfault observed only in this case
def test_minimal_segfault():
    t1 = example_input()
    s1 = t1.to_sparse()
    t2 = example_input()
    s2 = t2.to_sparse(1)
