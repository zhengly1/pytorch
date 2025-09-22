/*
 * PyTorch wrapper implementation for ULP-aware CUTLASS GEMM
 */

#include "UlpGemm.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <c10/util/Exception.h>

namespace at::native::ulp_gemm {

bool is_ulp_gemm_available(const Tensor& mat1, const Tensor& mat2) {
  // Check environment variable
  if (!should_use_ulp_gemm()) {
    return false;
  }
  
  // Check device support
  if (!is_device_supported()) {
    return false;
  }
  
  // Check tensor properties
  if (!mat1.is_cuda() || !mat2.is_cuda()) {
    return false;
  }
  
  // Check supported data types
  auto dtype = mat1.scalar_type();
  if (dtype != at::ScalarType::Float && 
      dtype != at::ScalarType::Half && 
      dtype != at::ScalarType::BFloat16) {
    return false;
  }
  
  // Check that both tensors have the same dtype
  if (mat1.scalar_type() != mat2.scalar_type()) {
    return false;
  }
  
  // Check tensor dimensions
  if (mat1.dim() != 2 || mat2.dim() != 2) {
    return false;
  }
  
  // Check size constraints (typical CUTLASS requirements)
  auto mat1_sizes = mat1.sizes();
  auto mat2_sizes = mat2.sizes();
  
  // Check alignment requirements (CUTLASS typically requires alignment)
  if (mat1_sizes[1] != mat2_sizes[0]) {
    return false;
  }
  
  // Basic size checks - avoid very small matrices where overhead dominates
  if (mat1_sizes[0] < 16 || mat1_sizes[1] < 16 || mat2_sizes[1] < 16) {
    return false;
  }
  
  return true;
}

void ulp_gemm_dispatch(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& bias,
    Tensor& result,
    const Scalar& alpha,
    const Scalar& beta,
    bool transpose_mat1,
    bool transpose_mat2) {
  
  TORCH_CHECK(is_ulp_gemm_available(mat1, mat2), 
              "ULP GEMM not available for the given tensors");
  
  auto mat1_sizes = mat1.sizes();
  auto mat2_sizes = mat2.sizes();
  
  int64_t m = transpose_mat1 ? mat1_sizes[1] : mat1_sizes[0];
  int64_t k = transpose_mat1 ? mat1_sizes[0] : mat1_sizes[1];
  int64_t n = transpose_mat2 ? mat2_sizes[0] : mat2_sizes[1];
  
  TORCH_CHECK(k == (transpose_mat2 ? mat2_sizes[1] : mat2_sizes[0]),
              "Matrix dimensions must be compatible for multiplication");
  
  // Ensure result tensor is properly sized
  if (result.numel() == 0) {
    result.resize_({m, n});
  }
  
  TORCH_CHECK(result.size(0) == m && result.size(1) == n,
              "Result tensor has incorrect dimensions");
  
  // Dispatch based on data type
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      mat1.scalar_type(),
      "ulp_gemm_dispatch",
      [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        
        // Get tensor strides
        int64_t mat1_ld = transpose_mat1 ? mat1.stride(0) : mat1.stride(1);
        int64_t mat2_ld = transpose_mat2 ? mat2.stride(0) : mat2.stride(1);
        int64_t result_ld = result.stride(0);
        
        // Handle bias by adding to result if beta != 0
        if (beta.toComplexDouble() != 0.0 && bias.numel() > 0) {
          result.copy_(bias);
        } else if (beta.toComplexDouble() == 0.0) {
          result.zero_();
        }
        
        ulp_gemm_impl<scalar_t>(
            transpose_mat1,
            transpose_mat2,
            m, n, k,
            alpha.to<opmath_t>(),
            mat1.const_data_ptr<scalar_t>(),
            mat1_ld,
            mat2.const_data_ptr<scalar_t>(),
            mat2_ld,
            beta.to<opmath_t>(),
            result.mutable_data_ptr<scalar_t>(),
            result_ld
        );
      });
}

} // namespace at::native::ulp_gemm