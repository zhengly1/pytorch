/*
 * PyTorch wrapper for ULP-aware CUTLASS GEMM
 * Provides integration with PyTorch's CUDA BLAS infrastructure
 */

#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include "cutlass_extensions/ulp_gemm_interface.h"

namespace at::native::ulp_gemm {

/// Check if ULP GEMM is available for the given data type and device
bool is_ulp_gemm_available(const Tensor& mat1, const Tensor& mat2);

/// Dispatch ULP GEMM based on tensor data types
void ulp_gemm_dispatch(
    const Tensor& mat1,
    const Tensor& mat2, 
    const Tensor& bias,
    Tensor& result,
    const Scalar& alpha,
    const Scalar& beta,
    bool transpose_mat1,
    bool transpose_mat2);

/// Main ULP GEMM implementation to replace cuBLAS in addmm_out_cuda_impl
template <typename scalar_t>
void ulp_gemm_impl(
    bool transpose_mat1,
    bool transpose_mat2,
    int64_t m,
    int64_t n,
    int64_t k,
    at::opmath_type<scalar_t> alpha_val,
    const scalar_t* mat1_ptr,
    int64_t mat1_ld,
    const scalar_t* mat2_ptr,
    int64_t mat2_ld,
    at::opmath_type<scalar_t> beta_val,
    scalar_t* result_ptr,
    int64_t result_ld) {

  auto stream = at::cuda::getCurrentCUDAStream();
  
  int status = -1;
  
  if constexpr (std::is_same_v<scalar_t, float>) {
    status = ulp_gemm_f32(
        transpose_mat1, transpose_mat2,
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
        static_cast<float>(alpha_val),
        mat1_ptr, static_cast<int>(mat1_ld),
        mat2_ptr, static_cast<int>(mat2_ld),
        static_cast<float>(beta_val),
        result_ptr, static_cast<int>(result_ld),
        stream);
  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
    status = ulp_gemm_f16(
        transpose_mat1, transpose_mat2,
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
        static_cast<float>(alpha_val),
        reinterpret_cast<const cutlass::half_t*>(mat1_ptr), static_cast<int>(mat1_ld),
        reinterpret_cast<const cutlass::half_t*>(mat2_ptr), static_cast<int>(mat2_ld),
        static_cast<float>(beta_val),
        reinterpret_cast<cutlass::half_t*>(result_ptr), static_cast<int>(result_ld),
        stream);
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    status = ulp_gemm_bf16(
        transpose_mat1, transpose_mat2,
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
        static_cast<float>(alpha_val),
        reinterpret_cast<const cutlass::bfloat16_t*>(mat1_ptr), static_cast<int>(mat1_ld),
        reinterpret_cast<const cutlass::bfloat16_t*>(mat2_ptr), static_cast<int>(mat2_ld),
        static_cast<float>(beta_val),
        reinterpret_cast<cutlass::bfloat16_t*>(result_ptr), static_cast<int>(result_ld),
        stream);
  } else {
    TORCH_CHECK(false, "ULP GEMM not supported for data type: ", typeid(scalar_t).name());
  }
  
  TORCH_CHECK(status == 0, "ULP GEMM failed with status: ", status);
}

/// Check if ULP GEMM should be used (environment variable or compile-time flag)
inline bool should_use_ulp_gemm() {
  static const char* env_value = std::getenv("PYTORCH_USE_ULP_GEMM");
  return env_value != nullptr && strcmp(env_value, "1") == 0;
}

/// Check device compute capability for ULP GEMM support
inline bool is_device_supported() {
  auto device_props = at::cuda::getCurrentDeviceProperties();
  // Require compute capability 8.0+ for Tensor Core operations
  return device_props->major >= 8;
}

} // namespace at::native::ulp_gemm