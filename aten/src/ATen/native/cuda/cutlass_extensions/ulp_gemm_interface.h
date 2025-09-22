/*
 * High-level interface for ULP-aware CUTLASS GEMM
 * Provides convenient functions to dispatch ULP GEMM operations
 */

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#include "gemm/kernel/ulp_gemm.h"

namespace cutlass {
namespace gemm {
namespace device {

////////////////////////////////////////////////////////////////////////////////

/// ULP-aware GEMM device operation aliases
template <typename ElementA, typename ElementB, typename ElementC>
using UlpGemmDevice = UlpGemm<
  ElementA, cutlass::layout::RowMajor,
  ElementB, cutlass::layout::RowMajor, 
  ElementC, cutlass::layout::RowMajor,
  float  // Use float accumulator for precision
>;

} // namespace device
} // namespace gemm
} // namespace cutlass

// C-style interface for PyTorch integration
extern "C" {

/// Float32 ULP GEMM interface
int ulp_gemm_f32(
    bool transpose_a,
    bool transpose_b,
    int m,
    int n, 
    int k,
    float alpha,
    const float* a_ptr,
    int lda,
    const float* b_ptr,
    int ldb,
    float beta,
    float* c_ptr,
    int ldc,
    cudaStream_t stream);

/// Float16 ULP GEMM interface  
int ulp_gemm_f16(
    bool transpose_a,
    bool transpose_b,
    int m,
    int n,
    int k,
    float alpha,
    const cutlass::half_t* a_ptr,
    int lda,
    const cutlass::half_t* b_ptr,
    int ldb,
    float beta,
    cutlass::half_t* c_ptr,
    int ldc,
    cudaStream_t stream);

/// BFloat16 ULP GEMM interface
int ulp_gemm_bf16(
    bool transpose_a,
    bool transpose_b,
    int m,
    int n,
    int k,
    float alpha,
    const cutlass::bfloat16_t* a_ptr,
    int lda,
    const cutlass::bfloat16_t* b_ptr,
    int ldb,
    float beta,
    cutlass::bfloat16_t* c_ptr,
    int ldc,
    cudaStream_t stream);

} // extern "C"