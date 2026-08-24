#ifndef MLLM_QNN_ACTIVATION_SCALE_OVERRIDE_HPP
#define MLLM_QNN_ACTIVATION_SCALE_OVERRIDE_HPP

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "Tensor.hpp"

namespace mllm {

// mllm input_scale/output_scale tensors store a symmetric absolute range,
// rather than a quantization step. Per-tensor overrides take precedence over
// the optional uniform scale factor.
inline std::optional<float> qnnActivationScaleOverride(
    const std::string &tensor_name, float current_value) {
    const char *raw = std::getenv("MLLM_QWEN_ACTIVATION_SCALE_OVERRIDES");
    if (raw != nullptr && raw[0] != '\0') {
        const std::string_view entries(raw);
        size_t begin = 0;
        while (begin <= entries.size()) {
            size_t end = entries.find(',', begin);
            if (end == std::string_view::npos) end = entries.size();
            const std::string_view entry = entries.substr(begin, end - begin);
            const size_t equals = entry.find('=');
            if (equals != std::string_view::npos
                && entry.substr(0, equals) == tensor_name) {
                const std::string value_text(entry.substr(equals + 1));
                char *parse_end = nullptr;
                errno = 0;
                const float value = std::strtof(value_text.c_str(), &parse_end);
                if (errno == 0 && parse_end != value_text.c_str()
                    && *parse_end == '\0' && std::isfinite(value)
                    && value > 0.0F) {
                    return value;
                }
                return std::nullopt;
            }
            if (end == entries.size()) break;
            begin = end + 1;
        }
    }

    const char *factor_raw =
        std::getenv("MLLM_QWEN_ACTIVATION_SCALE_FACTOR");
    if (factor_raw != nullptr && factor_raw[0] != '\0') {
        char *parse_end = nullptr;
        errno = 0;
        const float factor = std::strtof(factor_raw, &parse_end);
        if (errno == 0 && parse_end != factor_raw && *parse_end == '\0'
            && std::isfinite(factor) && factor > 0.0F
            && std::isfinite(current_value) && current_value > 0.0F) {
            return current_value * factor;
        }
    }
    return std::nullopt;
}

inline void qnnSetPrivateActivationScale(Tensor &scale, float value) {
    // ParamLoader can back loaded parameters with a read-only mmap. Rebind the
    // scalar to private storage instead of mutating checkpoint memory.
    const std::shared_ptr<float> writable = std::make_shared<float>(value);
    scale.setHostPtr(writable.get(), writable);
}

} // namespace mllm

#endif // MLLM_QNN_ACTIVATION_SCALE_OVERRIDE_HPP
