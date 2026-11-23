// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "mllm/core/DataTypes.hpp"
#include "mllm/core/Tensor.hpp"

namespace mllm::qnn {

struct QNNSequenceTilePlan {
  size_t axis = 0;
  int32_t tile_count = 0;
  Tensor::shape_t compiled_shape;
};

inline QNNSequenceTilePlan planQnnSequenceTiling(const Tensor::shape_t& logical_shape, const Tensor::shape_t& compiled_shape,
                                                 int32_t configured_tile_sequence) {
  if (configured_tile_sequence <= 0) { throw std::invalid_argument("QNN sequence tile must be positive"); }
  if (logical_shape.empty() || logical_shape.size() != compiled_shape.size()) {
    throw std::invalid_argument("QNN sequence tiling requires equal nonzero graph-boundary ranks");
  }

  size_t sequence_axis = logical_shape.size();
  for (size_t axis = 0; axis < logical_shape.size(); ++axis) {
    if (logical_shape[axis] <= 0 || compiled_shape[axis] <= 0) {
      throw std::invalid_argument("QNN sequence tiling requires positive graph-boundary dimensions");
    }
    if (logical_shape[axis] == compiled_shape[axis]) { continue; }
    if (sequence_axis != logical_shape.size()) {
      throw std::invalid_argument("QNN sequence tiling requires exactly one expanded boundary dimension");
    }
    sequence_axis = axis;
  }

  if (sequence_axis == logical_shape.size()) {
    throw std::invalid_argument("QNN sequence tiling requires one expanded boundary dimension");
  }
  const int32_t compiled_rows = compiled_shape[sequence_axis];
  const int32_t logical_rows = logical_shape[sequence_axis];
  // Packed NHWC graph boundaries can keep the token columns inside each copied row.
  const bool packed_tokens = sequence_axis == 1 && compiled_shape.size() == 4
                             && configured_tile_sequence % compiled_rows == 0
                             && compiled_shape[2] == configured_tile_sequence / compiled_rows;
  if ((compiled_rows != configured_tile_sequence && !packed_tokens) || logical_rows <= compiled_rows
      || logical_rows % compiled_rows != 0) {
    throw std::invalid_argument("QNN logical sequence is not an exact multiple of the compiled tile");
  }

  return {
      .axis = sequence_axis,
      .tile_count = logical_rows / compiled_rows,
      .compiled_shape = compiled_shape,
  };
}

inline void copyQnnSequenceTile(Tensor& destination, int32_t destination_sequence, const Tensor& source,
                                int32_t source_sequence, int32_t sequence_count, size_t sequence_axis) {
  if (destination.isNil() || source.isNil() || destination.dtype() != source.dtype() || destination.rank() != source.rank()
      || sequence_axis >= source.rank() || sequence_count <= 0 || destination_sequence < 0 || source_sequence < 0
      || !destination.isContiguous() || !source.isContiguous()) {
    throw std::invalid_argument("incompatible QNN sequence-tile copy");
  }
  const auto destination_shape = destination.shape();
  const auto source_shape = source.shape();
  for (size_t axis = 0; axis < source_shape.size(); ++axis) {
    if (axis != sequence_axis && destination_shape[axis] != source_shape[axis]) {
      throw std::invalid_argument("QNN sequence-tile copy has incompatible non-sequence dimensions");
    }
  }
  if (destination_sequence + sequence_count > destination_shape[sequence_axis]
      || source_sequence + sequence_count > source_shape[sequence_axis]) {
    throw std::out_of_range("QNN sequence-tile copy exceeds a graph boundary");
  }
  if (lanesOfType(source.dtype()) != 1) {
    throw std::invalid_argument("QNN sequence tiling only supports scalar graph-boundary dtypes");
  }

  size_t outer = 1;
  size_t inner = 1;
  for (size_t axis = 0; axis < sequence_axis; ++axis) { outer *= static_cast<size_t>(source_shape[axis]); }
  for (size_t axis = sequence_axis + 1; axis < source_shape.size(); ++axis) {
    inner *= static_cast<size_t>(source_shape[axis]);
  }
  const size_t element_bytes = bytesOfType(source.dtype());
  const size_t copy_bytes = static_cast<size_t>(sequence_count) * inner * element_bytes;
  auto* destination_bytes = destination.ptr<uint8_t>();
  const auto* source_bytes = source.ptr<uint8_t>();
  for (size_t outer_index = 0; outer_index < outer; ++outer_index) {
    const size_t destination_offset =
        (outer_index * static_cast<size_t>(destination_shape[sequence_axis]) + static_cast<size_t>(destination_sequence))
        * inner * element_bytes;
    const size_t source_offset =
        (outer_index * static_cast<size_t>(source_shape[sequence_axis]) + static_cast<size_t>(source_sequence)) * inner
        * element_bytes;
    std::memcpy(destination_bytes + destination_offset, source_bytes + source_offset, copy_bytes);
  }
}

}  // namespace mllm::qnn
