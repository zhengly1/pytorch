/*
 * Copyright (c) 2024
 * Custom CUTLASS epilogue with ULP perturbation for left operand
 */

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/epilogue/thread/linear_combination_params.h"
#include "ulp_perturbation.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Custom linear combination with ULP perturbation applied to left operand during accumulation
///
/// D = alpha * accumulator + beta * source, where accumulator has ULP perturbation applied
///
template <
  typename ElementOutput_,                             ///< Data type used to load and store tensors
  int Count,                                           ///< Number of elements computed per operation.
  typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
  typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
  ScaleType::Kind Scale = ScaleType::Default,          ///< Control Alpha and Beta scaling
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
  typename ElementSource_ = ElementOutput_
>
class LinearCombinationUlpPerturbation {
public:

  using ElementOutput = ElementOutput_;
  using ElementSource = ElementSource_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementScalar = ElementCompute;
  using ElementC = ElementSource_;
  using ElementD = ElementOutput_;

  static int const kCount = Count;
  static const ScaleType::Kind kScale = Scale;
  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentSource = Array<ElementSource, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using FragmentCompute = Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;

  /// Host-constructable parameters structure
  struct Params 
  {
    ElementCompute alpha;                         ///< scales accumulators
    ElementCompute beta;                          ///< scales source tensor
    ElementCompute const *alpha_ptr;              ///< pointer to accumulator scalar - if not null, loads it from memory
    ElementCompute const *beta_ptr;               ///< pointer to source scalar - if not null, loads it from memory
    ElementCompute const* const* alpha_ptr_array; ///< array of pointers to accumulator scalar per group/batch
    ElementCompute const* const* beta_ptr_array;  ///< array of pointers to source scalar per group/batch

    CUTLASS_HOST_DEVICE
    Params():
      alpha(ElementCompute(1)),
      beta(ElementCompute(0)),
      alpha_ptr(nullptr),
      beta_ptr(nullptr),
      alpha_ptr_array(nullptr),
      beta_ptr_array(nullptr) { }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute alpha,
      ElementCompute beta
    ):
      alpha(alpha), beta(beta),
      alpha_ptr(nullptr), beta_ptr(nullptr),
      alpha_ptr_array(nullptr), beta_ptr_array(nullptr) { }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute alpha
    ):
      alpha(alpha), beta(0),
      alpha_ptr(nullptr), beta_ptr(nullptr),
      alpha_ptr_array(nullptr), beta_ptr_array(nullptr) { }
  };

private:

  //
  // Data members
  //

  ElementCompute alpha_;
  ElementCompute beta_;

public:

  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  LinearCombinationUlpPerturbation(Params const &params, int group_idx = 0) {
    if (params.alpha_ptr_array != nullptr && params.alpha_ptr_array[group_idx] != nullptr) {
      alpha_ = *(params.alpha_ptr_array[group_idx]);
    }
    else if (params.alpha_ptr != nullptr) {
      alpha_ = *params.alpha_ptr;
    }
    else {
      alpha_ = params.alpha;
    }
    if (params.beta_ptr_array != nullptr && params.beta_ptr_array[group_idx] != nullptr) {
      beta_ = *(params.beta_ptr_array[group_idx]);
    }
    else if (params.beta_ptr != nullptr) {
      beta_ = *params.beta_ptr;
    }
    else {
      beta_ = params.beta;
    }
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    if (Scale == ScaleType::NoBetaScaling) return true;
    if (Scale == ScaleType::OnlyAlphaScaling) return false;
    if (Scale == ScaleType::Nothing) return false;
    return beta_ != ElementCompute(0);
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }
  }

  /// Apply ULP perturbation to accumulator elements before linear combination
  template<typename Element>
  CUTLASS_HOST_DEVICE
  void apply_ulp_perturbation(Array<Element, kCount> &accumulator) const {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      accumulator[i] = ulp_perturb_down(accumulator[i]);
    }
  }

  /// Computes linear scaling with source: D = alpha * ULP_PERTURB(accumulator) + beta * source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
      FragmentAccumulator const &accumulator,
      FragmentSource const &source) const {

    // Convert source to internal compute numeric type
    NumericArrayConverter<ElementCompute, ElementSource, kCount, Round> source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    FragmentCompute converted_source = source_converter(source);
    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Apply ULP perturbation to the accumulator (left operand in the final addition)
    apply_ulp_perturbation(converted_accumulator);

    if (Scale == ScaleType::Nothing)
      return destination_converter(converted_accumulator);

    // Perform binary operations
    FragmentCompute intermediate;

    multiplies<FragmentCompute> mul_add_source;
    multiply_add<FragmentCompute> mul_add_accumulator;

    if (Scale == ScaleType::NoBetaScaling)
      intermediate = converted_source;
    else
      intermediate = mul_add_source(beta_, converted_source);                             // X =  beta * C + uniform

    intermediate = mul_add_accumulator(alpha_, converted_accumulator, intermediate);    // D = alpha * ULP_PERTURB(Accum) + X

    return destination_converter(intermediate);
  }

  /// Computes linear scaling: D = alpha * ULP_PERTURB(accumulator)
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
      FragmentAccumulator const &accumulator) const {

    // Convert source to internal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Apply ULP perturbation to the accumulator (left operand)
    apply_ulp_perturbation(converted_accumulator);

    if (Scale == ScaleType::Nothing)
      return destination_converter(converted_accumulator);

    // Perform binary operations
    FragmentCompute intermediate;
    multiplies<FragmentCompute> mul_accumulator;

    intermediate = mul_accumulator(alpha_, converted_accumulator);    // D = alpha * ULP_PERTURB(Accum)

    return destination_converter(intermediate);
  }

  //
  // Specializations for scalar (for use with cute::collective::DefaultEpilogue)
  //
  CUTLASS_HOST_DEVICE
  ElementD operator()(ElementAccumulator const accumulator, ElementC const source) const {
    // Convert everything to Compute type, do compute, and then store to output type
    NumericConverter<ElementCompute, ElementAccumulator, Round> accumulator_converter;
    [[maybe_unused]] NumericConverter<ElementCompute, ElementC, Round> source_converter;
    NumericConverter<ElementD, ElementCompute, Round> destination_converter;

    // Convert and apply ULP perturbation to accumulator
    ElementCompute converted_accumulator = accumulator_converter(accumulator);
    converted_accumulator = ulp_perturb_down(converted_accumulator);

    if constexpr (Scale == ScaleType::Nothing) {
      return destination_converter(converted_accumulator);
    }

    // Perform binary operations
    ElementCompute intermediate;
    multiplies<ElementCompute> multiply;
    multiply_add<ElementCompute> madd;

    if constexpr (Scale == ScaleType::NoBetaScaling) {
      intermediate = source_converter(source);
    }
    else {
      intermediate = multiply(beta_, source);                            // X =  beta * C + uniform
    }

    intermediate = madd(alpha_, converted_accumulator, intermediate);   // alpha * ULP_PERTURB(Accum) + X

    return destination_converter(intermediate);
  }

  CUTLASS_HOST_DEVICE
  ElementD operator()(ElementAccumulator const accumulator) const {
    // Convert everything to Compute type, do compute, and then store to output type
    NumericConverter<ElementCompute, ElementAccumulator, Round> accumulator_converter;
    NumericConverter<ElementD, ElementCompute, Round> destination_converter;

    // Convert and apply ULP perturbation to accumulator
    ElementCompute converted_accumulator = accumulator_converter(accumulator);
    converted_accumulator = ulp_perturb_down(converted_accumulator);

    if constexpr (Scale == ScaleType::Nothing) {
      return destination_converter(converted_accumulator);
    }

    ElementCompute intermediate;
    multiplies<ElementCompute> multiply;

    intermediate = multiply(alpha_, converted_accumulator);             // alpha * ULP_PERTURB(Accum)

    return destination_converter(intermediate);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////