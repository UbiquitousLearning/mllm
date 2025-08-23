//===----------------------------------------------------------------------===//
// Print Stuff
//===----------------------------------------------------------------------===//
#include <fmt/ranges.h>
#include <vector>
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/core/ParameterFile.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/engine/MemoryManager.hpp"

namespace fmt {
template<>
struct formatter<mllm::DataTypes> {
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
  template<typename FormatContext>
  auto format(const mllm::DataTypes& dtype, FormatContext& ctx) const {
    auto out = ctx.out();
    out = fmt::format_to(out, "{}", mllm::nameOfType(dtype));
    return out;
  }
};

template<>
struct formatter<mllm::DeviceTypes> {
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
  template<typename FormatContext>
  auto format(const mllm::DeviceTypes& device, FormatContext& ctx) const {
    auto out = ctx.out();
    out = fmt::format_to(out, "{}", mllm::deviceTypes2Str(device));
    return out;
  }
};

template<>
struct formatter<std::vector<int32_t>> {
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
  template<typename FormatContext>
  auto format(const std::vector<int32_t>& vec, FormatContext& ctx) const {
    auto out = ctx.out();
    *out++ = '[';
    for (size_t i = 0; i < vec.size(); ++i) {
      if (i > 0) {
        *out++ = ',';
        *out++ = ' ';
      }
      out = fmt::format_to(out, "{}", vec[i]);
    }
    *out++ = ']';
    return out;
  }
};

template<>
struct formatter<mllm::Tensor> {
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

  template<typename FormatContext>
  auto format(const mllm::Tensor& tensor, FormatContext& ctx) const {
    if (tensor.isNil()) { return fmt::format_to(ctx.out(), "Tensor(nil)"); }

    if (tensor.numel() == 0) {
      return fmt::format_to(ctx.out(), "tensor([], size=({}), dtype={})", fmt::join(tensor.shape(), ", "), tensor.dtype());
    }

    auto out = fmt::format_to(ctx.out(), "tensor(\n");
    out = printTensorData(tensor, out, 0, {});
    return fmt::format_to(out, ", dtype={}, device={})", tensor.dtype(), tensor.device());
  }

 private:
  template<typename OutputIt>
  OutputIt printTensorData(const mllm::Tensor& tensor, OutputIt out, int dim, const std::vector<int32_t>& indices) const {
    auto shape = tensor.shape();

    if (dim >= (int)shape.size()) { return printTensorValue(tensor, out, indices); }

    int32_t dim_size = shape[dim];
    *out++ = '[';

    // 获取Context中设置的最大元素数
    int max_elements_per_dim = mllm::Context::instance().getPrintMaxElementsPerDim();

    if (dim_size <= max_elements_per_dim) {
      for (int32_t i = 0; i < dim_size; ++i) {
        if (i > 0) {
          *out++ = ',';
          if (dim != (int)shape.size() - 1) {
            *out++ = '\n';
            for (int j = 0; j <= dim; ++j) *out++ = ' ';
          } else {
            *out++ = ' ';
          }
        }
        std::vector<int32_t> new_indices = indices;
        new_indices.push_back(i);
        out = printTensorData(tensor, out, dim + 1, new_indices);
      }
    } else {
      const int SHOW_ELEMENTS = max_elements_per_dim / 2;

      for (int32_t i = 0; i < SHOW_ELEMENTS; ++i) {
        if (i > 0) {
          *out++ = ',';
          if (dim != (int)shape.size() - 1) {
            *out++ = '\n';
            for (int j = 0; j <= dim; ++j) *out++ = ' ';
          } else {
            *out++ = ' ';
          }
        }
        std::vector<int32_t> new_indices = indices;
        new_indices.push_back(i);
        out = printTensorData(tensor, out, dim + 1, new_indices);
      }

      *out++ = ',';
      if (dim != (int)shape.size() - 1) {
        *out++ = '\n';
        for (int j = 0; j <= dim; ++j) *out++ = ' ';
      } else {
        *out++ = ' ';
      }
      *out++ = '.';
      *out++ = '.';
      *out++ = '.';

      for (int32_t i = dim_size - SHOW_ELEMENTS; i < dim_size; ++i) {
        *out++ = ',';
        if (dim != (int)shape.size() - 1) {
          *out++ = '\n';
          for (int j = 0; j <= dim; ++j) *out++ = ' ';
        } else {
          *out++ = ' ';
        }
        std::vector<int32_t> new_indices = indices;
        new_indices.push_back(i);
        out = printTensorData(tensor, out, dim + 1, new_indices);
      }
    }

    *out++ = ']';
    return out;
  }

  template<typename OutputIt>
  OutputIt printTensorValue(const mllm::Tensor& tensor, OutputIt out, const std::vector<int32_t>& indices) const {
    // 获取Context中设置的打印精度
    int precision = mllm::Context::instance().getPrintPrecision();

    switch (tensor.dtype()) {
      case mllm::kFloat32:
        return fmt::format_to(out, "{:.{}e}", tensor.constAt<mllm::mllm_fp32_t>(const_cast<std::vector<int32_t>&>(indices)),
                              precision);
      case mllm::kFloat16:
        return fmt::format_to(
            out, "{:.{}e}",
            static_cast<mllm::mllm_fp32_t>(tensor.constAt<mllm::mllm_fp16_t>(const_cast<std::vector<int32_t>&>(indices))),
            precision);
      case mllm::kInt32:
        return fmt::format_to(out, "{}", tensor.constAt<mllm::mllm_int32_t>(const_cast<std::vector<int32_t>&>(indices)));
      case mllm::kInt16:
        return fmt::format_to(out, "{}", tensor.constAt<mllm::mllm_int16_t>(const_cast<std::vector<int32_t>&>(indices)));
      case mllm::kInt8:
        return fmt::format_to(
            out, "{}",
            static_cast<mllm::mllm_int32_t>(tensor.constAt<mllm::mllm_int8_t>(const_cast<std::vector<int32_t>&>(indices))));
      case mllm::kUInt8:
        return fmt::format_to(
            out, "{}",
            static_cast<mllm::mllm_int32_t>(tensor.constAt<mllm::mllm_uint8_t>(const_cast<std::vector<int32_t>&>(indices))));
      case mllm::kInt64:
        return fmt::format_to(out, "{}", tensor.constAt<mllm::mllm_int64_t>(const_cast<std::vector<int32_t>&>(indices)));
      case mllm::kUInt32:
        return fmt::format_to(out, "{}", tensor.constAt<mllm::mllm_uint32_t>(const_cast<std::vector<int32_t>&>(indices)));
      case mllm::kUInt64:
        return fmt::format_to(out, "{}", tensor.constAt<mllm::mllm_uint64_t>(const_cast<std::vector<int32_t>&>(indices)));
      default: return fmt::format_to(out, "?");
    }
  }
};

template<>
struct formatter<mllm::ParameterFile::ptr_t> {
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
  template<typename FormatContext>
  auto format(const mllm::ParameterFile::ptr_t& params, FormatContext& ctx) const {
    if (!params) { return fmt::format_to(ctx.out(), "ParameterFile[nullptr]"); }

    static std::unordered_map<mllm::ModelFileVersion, std::string> version_names = {
        {mllm::ModelFileVersion::kUserTemporary, "UserTemporary"},
        {mllm::ModelFileVersion::kV1, "V1"},
        {mllm::ModelFileVersion::kV2, "V2"},
    };
    std::string version_str = "Unknown";
    if (auto it = version_names.find(params->version()); it != version_names.end()) {
      version_str = it->second;
    } else {
      version_str = fmt::format("{}", static_cast<int>(params->version()));
    }

    struct ParamInfo {
      std::string name;
      std::string shape;
      std::string dtype;
    };

    std::vector<ParamInfo> param_infos;
    for (auto it = params->begin(); it != params->end(); ++it) {
      auto tensor = params->pull(it->first);
      param_infos.push_back({it->first, fmt::format("{}", tensor.shape()), fmt::format("{}", tensor.dtype())});
    }

    std::sort(param_infos.begin(), param_infos.end(), [](const ParamInfo& a, const ParamInfo& b) { return a.name < b.name; });

    const size_t MAX_SHOW = 2048;
    const size_t total_params = param_infos.size();
    auto out = fmt::format_to(ctx.out(), "ParameterFile[{} params, version={}]:\n", total_params, version_str);

    size_t max_name_width = 4;   // "name"
    size_t max_shape_width = 5;  // "shape"
    size_t max_dtype_width = 5;  // "dtype"

    for (const auto& info : param_infos) {
      max_name_width = std::max(max_name_width, info.name.length());
      max_shape_width = std::max(max_shape_width, info.shape.length());
      max_dtype_width = std::max(max_dtype_width, info.dtype.length());
    }

    max_name_width = std::min(max_name_width + 2, size_t(60));
    max_shape_width = std::min(max_shape_width + 2, size_t(30));
    max_dtype_width = std::min(max_dtype_width + 2, size_t(20));

    auto make_separator = [](size_t width, char c) { return std::string(width, c); };

    std::string name_sep = make_separator(max_name_width, '-');
    std::string shape_sep = make_separator(max_shape_width, '-');
    std::string dtype_sep = make_separator(max_dtype_width, '-');

    out = fmt::format_to(out, "+-{}-+-{}-+-{}-+\n", name_sep, shape_sep, dtype_sep);
    out = fmt::format_to(out, "| {:<{}} | {:<{}} | {:<{}} |\n", "name", max_name_width, "shape", max_shape_width, "dtype",
                         max_dtype_width);
    out = fmt::format_to(out, "+={}=+={}=+={}=+\n", make_separator(max_name_width, '='), make_separator(max_shape_width, '='),
                         make_separator(max_dtype_width, '='));

    for (size_t i = 0; i < std::min(MAX_SHOW, total_params); ++i) {
      const auto& info = param_infos[i];
      out = fmt::format_to(out, "| {:<{}} | {:<{}} | {:<{}} |\n", info.name, max_name_width, info.shape, max_shape_width,
                           info.dtype, max_dtype_width);
    }

    out = fmt::format_to(out, "+-{}-+-{}-+-{}-+\n", name_sep, shape_sep, dtype_sep);

    if (total_params > MAX_SHOW) {
      size_t remaining_width = max_name_width + max_shape_width + max_dtype_width - 16;
      out = fmt::format_to(out, "| ... and {} more parameters {:<{}} |\n", total_params - MAX_SHOW, "", remaining_width);
      out = fmt::format_to(out, "+-{}-+-{}-+-{}-+\n", name_sep, shape_sep, dtype_sep);
    } else if (total_params == 0) {
      size_t empty_width = max_name_width + max_shape_width + max_dtype_width - 15;
      out = fmt::format_to(out, "| [No parameters] {:<{}} |\n", "", empty_width);
      out = fmt::format_to(out, "+-{}-+-{}-+-{}-+\n", name_sep, shape_sep, dtype_sep);
    }
    return out;
  }
};

template<typename T>
struct formatter<T, std::enable_if_t<std::is_base_of_v<mllm::nn::Module, T>, char>> : formatter<std::string> {
  auto format(const mllm::nn::Module& custom_module, format_context& ctx) const {
    std::stringstream ss;
    custom_module.__fmt_print(ss);
    return formatter<std::string>::format(ss.str(), ctx);
  }
};

template<>
struct formatter<mllm::MemoryManager::ptr_t> {
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
  template<typename FormatContext>
  auto format(const mllm::MemoryManager::ptr_t& params, FormatContext& ctx) const {
    params->report();
    return ctx.out();
  }
};

template<>
struct formatter<mllm::ir::IRContext::ptr_t> {
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
  template<typename FormatContext>
  auto format(const mllm::ir::IRContext::ptr_t& ir_ctx, FormatContext& ctx) const {
    auto out = ctx.out();

    auto dumpper = ::mllm::ir::IRPrinter();
    ir_ctx->topLevelOp()->dump(dumpper);

    out = fmt::format_to(out, "");
    return out;
  }
};

template<>
struct formatter<::mllm::test::AllCloseResult> {
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
  template<typename FormatContext>
  auto format(const mllm::test::AllCloseResult& result, FormatContext& ctx) const {
    auto out = ctx.out();
    out = fmt::format_to(out,
                         "AllCloseResult(is_close={}, total_elements={}, mismatched_elements={}, max_absolute_diff={:.6f}, "
                         "max_relative_diff={:.6f})",
                         result.is_close, result.total_elements, result.mismatched_elements, result.max_absolute_diff,
                         result.max_relative_diff);
    return out;
  }
};

template<>
struct formatter<::mllm::ConfigFile> {
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
  template<typename FormatContext>
  auto format(const mllm::ConfigFile& cfg, FormatContext& ctx) const {
    auto out = ctx.out();
    out = fmt::format_to(out, "{}", cfg.dump());
    return out;
  }
};
}  // namespace fmt

#define MLLM_MAJOR_VERSION = 2
#define MLLM_MINOR_VERSION = 0
#define MLLM_PATCH_VERSION = 0
#define MLLM_VERSION_STRING "2.0.0"
