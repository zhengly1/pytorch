/*
 * CUTLASS GEMM implementation with ULP perturbation - CUDA source
 */

#include "cutlass_gemm_ulp_perturbation.h"
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/layout/matrix.h>

namespace at::native::cutlass_gemm {

// Simple CUTLASS GEMM device configuration for different types
template<typename ElementType>
struct GemmDeviceConfig {
    static constexpr int kAlignment = 8;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using ElementAccumulator = float;
    
    // Use our custom epilogue with ULP perturbation
    using EpilogueOp = cutlass::epilogue::thread::LinearCombinationUlpPerturbation<
        ElementType,                                  // ElementOutput
        128 / cutlass::sizeof_bits<ElementType>::value, // Count
        ElementAccumulator,                           // ElementAccumulator
        ElementAccumulator                            // ElementCompute
    >;
    
    // Simple SIMT-based GEMM for compatibility
    using DeviceGemm = cutlass::gemm::device::Gemm<
        ElementType, LayoutA,
        ElementType, LayoutB,
        ElementType, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        cutlass::gemm::GemmShape<128, 128, 8>,
        cutlass::gemm::GemmShape<32, 64, 8>,
        cutlass::gemm::GemmShape<1, 1, 1>,
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2,
        kAlignment,
        kAlignment
    >;
};

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
    int64_t ldc) {
    
    // Map PyTorch types to CUTLASS types
    using CutlassType = typename pytorch_to_cutlass_type<scalar_t>::type;
    using DeviceGemm = typename GemmDeviceConfig<CutlassType>::DeviceGemm;
    using ElementAccumulator = typename GemmDeviceConfig<CutlassType>::ElementAccumulator;
    
    // Setup CUTLASS arguments
    cutlass::gemm::GemmCoord problem_size(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
    
    // Cast pointers to CUTLASS types
    const CutlassType* matrix_a = reinterpret_cast<const CutlassType*>(A);
    const CutlassType* matrix_b = reinterpret_cast<const CutlassType*>(B);
    CutlassType* matrix_c = reinterpret_cast<CutlassType*>(C);
    
    // Handle transpose: CUTLASS expects row-major A and column-major B
    // If we have different layouts, we may need to swap operands or adjust
    int64_t leading_dim_a = lda;
    int64_t leading_dim_b = ldb;
    int64_t leading_dim_c = ldc;
    
    typename DeviceGemm::Arguments arguments{
        problem_size,
        {matrix_a, leading_dim_a},
        {matrix_b, leading_dim_b},
        {matrix_c, leading_dim_c},
        {matrix_c, leading_dim_c},
        {static_cast<ElementAccumulator>(alpha), static_cast<ElementAccumulator>(beta)}
    };
    
    // Launch CUTLASS GEMM
    DeviceGemm gemm_op;
    
    // Check if the operation can be supported
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        TORCH_CHECK(false, "CUTLASS GEMM cannot implement the requested operation");
    }
    
    size_t workspace_size = DeviceGemm::get_workspace_size(arguments);
    
    // Allocate workspace if needed
    void* workspace = nullptr;
    if (workspace_size > 0) {
        workspace = c10::cuda::CUDACachingAllocator::raw_alloc(workspace_size);
    }
    
    status = gemm_op.initialize(arguments, workspace);
    if (status != cutlass::Status::kSuccess) {
        if (workspace) {
            c10::cuda::CUDACachingAllocator::raw_delete(workspace);
        }
        TORCH_CHECK(false, "CUTLASS GEMM initialization failed");
    }
    
    status = gemm_op();
    if (workspace) {
        c10::cuda::CUDACachingAllocator::raw_delete(workspace);
    }
    
    if (status != cutlass::Status::kSuccess) {
        TORCH_CHECK(false, "CUTLASS GEMM execution failed");
    }
}

// Explicit template instantiations
template void launch_cutlass_gemm_ulp_perturbation<float>(
    bool, bool, int64_t, int64_t, int64_t, float, const float*, int64_t,
    const float*, int64_t, float, float*, int64_t);

template void launch_cutlass_gemm_ulp_perturbation<c10::Half>(
    bool, bool, int64_t, int64_t, int64_t, float, const c10::Half*, int64_t,
    const c10::Half*, int64_t, float, c10::Half*, int64_t);

template void launch_cutlass_gemm_ulp_perturbation<c10::BFloat16>(
    bool, bool, int64_t, int64_t, int64_t, float, const c10::BFloat16*, int64_t,
    const c10::BFloat16*, int64_t, float, c10::BFloat16*, int64_t);

} // namespace at::native::cutlass_gemm