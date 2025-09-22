/*
 * Custom CUTLASS GEMM kernel with ULP perturbation
 * Simplified version focusing on the core ULP functionality
 */

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#include "../epilogue/ulp_linear_combination.h"

namespace cutlass {
namespace gemm {
namespace device {

////////////////////////////////////////////////////////////////////////////////

/// Simplified ULP-aware GEMM using standard CUTLASS device GEMM
template <
  typename ElementA_,
  typename LayoutA_,
  typename ElementB_,
  typename LayoutB_,
  typename ElementC_,
  typename LayoutC_,
  typename ElementAccumulator_ = ElementC_,
  typename ActivationFunctor_ = cutlass::epilogue::thread::Identity<ElementC_>
>
using UlpGemm = cutlass::gemm::device::Gemm<
  ElementA_,
  LayoutA_,
  ElementB_,
  LayoutB_,
  ElementC_,
  LayoutC_,
  ElementAccumulator_,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128, 256, 32>,
  cutlass::gemm::GemmShape<64, 64, 32>,
  cutlass::gemm::GemmShape<16, 8, 8>,
  cutlass::epilogue::thread::UlpLinearCombination<
    ElementC_,
    128 / cutlass::sizeof_bits<ElementC_>::value,
    ElementAccumulator_,
    ElementAccumulator_,
    cutlass::FloatRoundStyle::round_to_nearest,
    ActivationFunctor_
  >,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  3
>;

////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cutlass