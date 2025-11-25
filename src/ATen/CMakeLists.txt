# ATen XPU sources

file(GLOB xpu_cpp "xpu/*.cpp")
file(GLOB xpu_native_cpp "native/xpu/*.cpp" "native/sparse/*.cpp" "native/sparse/xpu/*.cpp" "native/nested/*.cpp" "native/nested/xpu/*.cpp" "native/transformers/*.cpp" "native/quantized/*.cpp" ${TORCH_ROOT}/aten/src/ATen/native/transformers/xpu/flash_attn/*.cpp)
file(GLOB xpu_sycl "native/xpu/sycl/*.cpp" "native/sparse/xpu/sycl/*.cpp" "native/nested/xpu/sycl/*.cpp" "native/transformers/sycl/*.cpp" "native/quantized/sycl/*.cpp")
file(GLOB xpu_sycltla "${TORCH_ROOT}/aten/src/ATen/native/transformers/xpu/flash_attn/sycltla/*.cpp")

list(APPEND ATen_XPU_CPP_SRCS ${xpu_cpp})
if(USE_ONEMKL_XPU)
  file(GLOB xpu_mkl "native/xpu/mkl/*.cpp")
  list(APPEND ATen_XPU_MKL_SRCS ${xpu_mkl})
endif()
list(APPEND ATen_XPU_NATIVE_CPP_SRCS ${xpu_native_cpp})
list(APPEND ATen_XPU_SYCL_SRCS ${xpu_sycl})
list(APPEND ATen_XPU_SYCLTLA_SRCS ${xpu_sycltla})

set(ATen_XPU_CPP_SRCS ${ATen_XPU_CPP_SRCS} PARENT_SCOPE)
set(ATen_XPU_MKL_SRCS ${ATen_XPU_MKL_SRCS} PARENT_SCOPE)
set(ATen_XPU_NATIVE_CPP_SRCS ${ATen_XPU_NATIVE_CPP_SRCS} PARENT_SCOPE)
set(ATen_XPU_SYCL_SRCS ${ATen_XPU_SYCL_SRCS} PARENT_SCOPE)
set(ATen_XPU_SYCLTLA_SRCS ${ATen_XPU_SYCLTLA_SRCS} PARENT_SCOPE)

# ATen XPU headers

macro(install_xpu_headers subdir)
  file(GLOB headers CONFIGURE_DEPENDS "${subdir}/*.h")
  if(headers)
    install(FILES ${headers} DESTINATION "${AT_INSTALL_INCLUDE_DIR}/ATen/${subdir}")
  endif()
endmacro()

set(XPU_HEADER_DIRS
  "xpu"
  "native/xpu"
  "native/xpu/sycl"
  "native/xpu/mkl"
  "native/nested/xpu"
  "native/nested/xpu/sycl"
  "native/quantized/xpu"
  "native/quantized/xpu/sycl"
  "native/sparse/xpu"
  "native/sparse/xpu/sycl"
  "native/transformers/xpu"
  "native/transformers/xpu/sycl"
)

foreach(subdir ${XPU_HEADER_DIRS})
  install_xpu_headers(${subdir})
endforeach()

if(xpu_ops_generated_headers)
  install(FILES ${xpu_ops_generated_headers} DESTINATION ${AT_INSTALL_INCLUDE_DIR}/ATen/ops)
endif()
