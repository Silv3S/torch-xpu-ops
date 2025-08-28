#pragma once
#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void nonzero_kernel(const Tensor& self, Tensor& out);
TORCH_XPU_API void nonzero_static_kernel(const Tensor& self, int64_t size, int64_t fill_value, Tensor& out);

} // namespace at::native::xpu
