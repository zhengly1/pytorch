/*
 * Custom CUTLASS epilogue with ULP perturbation
 * Applies ULP perturbation to left operand before addition operations
 */

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/functional.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/arch/memory.h>
#include <cutlass/arch/memory_sm75.h>

#include "ulp_utils.h"

namespace cutlass {
namespace epilogue {
namespace thread {

////////////////////////////////////////////////////////////////////////////////

/// Applies ULP perturbation to elements before linear combination
template <
  typename ElementOutput_,                             ///< Data type used to load and store tensors
  int Count,                                           ///< Number of elements computed per operation
  typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
  typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
  cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest,
  typename ActivationFunctor_ = cutlass::epilogue::thread::Identity<ElementCompute_>
>
class UlpLinearCombination {
public:

  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ActivationFunctor = ActivationFunctor_;

  static int const kCount = Count;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using ComputeFragment = Array<ElementCompute, kCount>;

  static cutlass::FloatRoundStyle const kRound = Round;

  /// Host-constructible parameters structure
  struct Params {
    ElementCompute alpha;                  ///< scales accumulators
    ElementCompute beta;                   ///< scales source tensor
    typename ActivationFunctor::Params activation;  ///< activation function parameters

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(): alpha(ElementCompute(1)), beta(ElementCompute(0)) { }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute alpha,
      ElementCompute beta
    ): alpha(alpha), beta(beta) { }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute alpha,
      ElementCompute beta,
      typename ActivationFunctor::Params activation
    ): alpha(alpha), beta(beta), activation(activation) { }
  };

private:

  //
  // Data members
  //

  ElementCompute alpha_;
  ElementCompute beta_;
  ActivationFunctor activation_;

public:

  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  UlpLinearCombination(Params const &params) {
    alpha_ = params.alpha;
    beta_ = params.beta;
    activation_ = ActivationFunctor(params.activation);
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return beta_ != ElementCompute(0);
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }

    if (k_partition != k_partition_count - 1) {
      // Set alpha to 1 for intermediate accumulations
      alpha_ = ElementCompute(1);
    }
  }

  /// Computes linear scaling: D = alpha * accumulator + beta * source
  /// with ULP perturbation applied to the accumulator (left operand)
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &accumulator,
    FragmentOutput const &source) const {

    // Convert to the computation data type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;
    NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round> source_converter;

    ComputeFragment converted_accumulator = accumulator_converter(accumulator);
    ComputeFragment converted_source = source_converter(source);

    // Apply ULP perturbation to accumulator (left operand) before computation
    ComputeFragment perturbed_accumulator;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      perturbed_accumulator[i] = cutlass::gemm::thread::perturb_ulp_down(converted_accumulator[i]);
    }

    // Compute linear combination with perturbed accumulator
    ComputeFragment intermediate;
    
    multiplies<ComputeFragment> mul_add_source;
    multiply_add<ComputeFragment> mul_add_accumulator;

    if (beta_ == ElementCompute(0)) {
      multiplies<ComputeFragment> mul_accumulator;
      intermediate = mul_accumulator(alpha_, perturbed_accumulator);
    } else {
      intermediate = mul_add_accumulator(beta_, converted_source, alpha_, perturbed_accumulator);
    }

    // Apply activation function
    intermediate = activation_(intermediate);

    // Convert to output numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    return destination_converter(intermediate);
  }

  /// Computes linear scaling: D = alpha * accumulator
  /// with ULP perturbation applied to the accumulator
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &accumulator) const {

    // Convert to the computation data type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    // Apply ULP perturbation to accumulator before computation
    ComputeFragment perturbed_accumulator;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      perturbed_accumulator[i] = cutlass::gemm::thread::perturb_ulp_down(converted_accumulator[i]);
    }

    // Compute linear combination with perturbed accumulator
    ComputeFragment intermediate;
    
    multiplies<ComputeFragment> mul_accumulator;
    intermediate = mul_accumulator(alpha_, perturbed_accumulator);

    // Apply activation function
    intermediate = activation_(intermediate);

    // Convert to output numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    return destination_converter(intermediate);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Convenience alias for linear combination with ULP perturbation
template <
  typename ElementOutput_,                             ///< Data type used to load and store tensors
  int Count,                                           ///< Number of elements computed per operation
  typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
  typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
  cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest
>
using UlpLinearCombinationRelu = UlpLinearCombination<
  ElementOutput_, Count, ElementAccumulator_, ElementCompute_, Round, 
  cutlass::epilogue::thread::ReLu<ElementCompute_>>;

/// Convenience alias for linear combination with ULP perturbation and GELU
template <
  typename ElementOutput_,                             ///< Data type used to load and store tensors
  int Count,                                           ///< Number of elements computed per operation
  typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
  typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
  cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest
>
using UlpLinearCombinationGelu = UlpLinearCombination<
  ElementOutput_, Count, ElementAccumulator_, ElementCompute_, Round, 
  cutlass::epilogue::thread::GELU<ElementCompute_>>;

////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass