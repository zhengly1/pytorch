/*
 * Custom CUTLASS GEMM implementation with ULP perturbation
 * Replaces cuBLAS API calls in PyTorch with CUTLASS implementation
 */

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>

// Include our custom epilogue
#include "epilogue/thread/linear_combination_ulp_perturbation.h"

namespace at::native::cutlass_gemm {

// Helper to map PyTorch scalar types to CUTLASS types
template<typename T>
struct pytorch_to_cutlass_type {
    using type = T;
};

template<>
struct pytorch_to_cutlass_type<c10::Half> {
    using type = cutlass::half_t;
};

template<>
struct pytorch_to_cutlass_type<c10::BFloat16> {
    using type = cutlass::bfloat16_t;
};

// Function to launch CUTLASS GEMM with ULP perturbation
template<typename scalar_t>
void launch_cutlass_gemm_ulp_perturbation(
    bool transpose_a,
    bool transpose_b,
    int64_t m,
    int64_t n,
    int64_t k,
    at::opmath_type<scalar_t> alpha,
    const scalar_t* A,
    int64_t lda,
    const scalar_t* B,
    int64_t ldb,
    at::opmath_type<scalar_t> beta,
    scalar_t* C,
    int64_t ldc);

// Explicit template instantiations for supported types
extern template void launch_cutlass_gemm_ulp_perturbation<float>(
    bool, bool, int64_t, int64_t, int64_t, float, const float*, int64_t,
    const float*, int64_t, float, float*, int64_t);

extern template void launch_cutlass_gemm_ulp_perturbation<c10::Half>(
    bool, bool, int64_t, int64_t, int64_t, float, const c10::Half*, int64_t,
    const c10::Half*, int64_t, float, c10::Half*, int64_t);

extern template void launch_cutlass_gemm_ulp_perturbation<c10::BFloat16>(
    bool, bool, int64_t, int64_t, int64_t, float, const c10::BFloat16*, int64_t,
    const c10::BFloat16*, int64_t, float, c10::BFloat16*, int64_t);

} // namespace at::native::cutlass_gemm