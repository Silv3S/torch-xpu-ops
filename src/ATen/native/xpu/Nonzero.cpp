#include <ATen/core/Tensor.h>
#include <ATen/ops/full.h>
#include <ATen/xpu/EmptyTensor.h>

#include <ATen/native/xpu/sycl/NonzeroKernel.h>

namespace at {
namespace native {

void nonzero_common_checks(const Tensor& self, Tensor& out) {
  TORCH_CHECK(
      self.numel() < std::numeric_limits<int>::max(),
      "nonzero is not supported for tensors with more than INT_MAX elements, \
      See https://github.com/pytorch/pytorch/issues/51871");
  TORCH_CHECK(
      out.dtype() == at::kLong,
      "Expected object of scalar type ",
      at::kLong,
      " as out, but got ",
      out.dtype());
  TORCH_CHECK(
      self.device() == out.device(),
      "expected self and out to be on the same device, but got out on ",
      out.device(),
      " and self on ",
      self.device());
  TORCH_CHECK(
      self.dim() <= XPU_MAX_TENSORINFO_DIMS,
      "nonzero is not supported for tensor with more than ",
      XPU_MAX_TENSORINFO_DIMS,
      " dimensions");
}

Tensor& nonzero_out_xpu(const Tensor& self, Tensor& out) {
  nonzero_common_checks(self, out);
  xpu::nonzero_kernel(self, out);
  return out;
}

Tensor nonzero_xpu(const Tensor& self) {
  Tensor out = at::detail::empty_xpu({0}, self.options().dtype(kLong));
  nonzero_out_xpu(self, out);
  return out;
}

Tensor& nonzero_static_out_xpu(
    const Tensor& self,
    int64_t size,
    int64_t fill_value,
    Tensor& out) {
  nonzero_common_checks(self, out);
  xpu::nonzero_static_kernel(self, size, fill_value, out);

  // Naive implementation to just trim/expand result from existing nonzero_kernel
  // TODO - remove this block and implement it in nonzero_static_kernel
  int64_t num_nonzero = out.size(0); 
  int64_t out_size = std::max(size, num_nonzero);
  auto filled = at::full({out_size, self.dim()}, fill_value, out.options());
  if (num_nonzero > 0) {
    filled.narrow(0, 0, num_nonzero).copy_(out);
  }
  out = filled.narrow(0, 0, size);
  // Naive implementation to just trim/expand result from existing nonzero_kernel

  return out;
}

Tensor nonzero_static_xpu(
    const Tensor& self,
    int64_t size,
    int64_t fill_value) {
  Tensor out = at::detail::empty_xpu({size, self.dim()}, self.options().dtype(kLong));
  nonzero_static_out_xpu(self, size, fill_value, out);
  return out;
}

} // namespace native
} // namespace at