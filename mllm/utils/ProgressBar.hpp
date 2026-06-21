// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <chrono>
#include <string>
#include <cstdint>

#include <fmt/core.h>
#include <fmt/chrono.h>
#include <fmt/color.h>

namespace mllm {

class ProgressBar {
 public:
  ProgressBar(const std::string& label, int total, int bar_width = 50,
              fmt::text_style label_style = fg(fmt::color::light_blue) | fmt::emphasis::bold,
              fmt::text_style bar_style = fg(fmt::color::green) | bg(fmt::color::dark_gray),
              fmt::text_style percent_style = fg(fmt::color::yellow) | fmt::emphasis::bold,
              fmt::text_style time_style = fg(fmt::color::cyan))
      : label_(label),
        total_(total),
        bar_width_(bar_width),
        start_time_(std::chrono::steady_clock::now()),
        label_style_(label_style),
        bar_style_(bar_style),
        percent_style_(percent_style),
        time_style_(time_style) {}

  void update(int current) {
    using namespace std::chrono;  // NOLINT
    auto now = steady_clock::now();
    auto elapsed = duration_cast<seconds>(now - start_time_);
    double progress = static_cast<double>(current) / total_;

    seconds remaining(0);
    if (current > 0 && current < total_) {
      double time_per_unit = elapsed.count() / static_cast<double>(current);
      remaining = seconds(static_cast<int64_t>(time_per_unit * (total_ - current)));
    }

    int pos = static_cast<int>(bar_width_ * progress);
    std::string bar = fmt::format("[{:{}}]",
                                  fmt::format("{:█^{}}{: <{}}", (pos > 0) ? "" : "", std::min(pos, bar_width_),
                                              (pos < bar_width_) ? ">" : "", (pos < bar_width_) ? bar_width_ - pos - 1 : 0),
                                  bar_width_);

    std::string styled_label = fmt::format(label_style_, "{:<12}", label_);
    std::string styled_bar = fmt::format(bar_style_, "{}", bar);
    std::string styled_percent = fmt::format(percent_style_, "{:>3}%", static_cast<int>(progress * 100));

    std::string time_info;
    if (current == 0) {
      time_info = " --:-- / --:-- ";
    } else if (current >= total_) {
      time_info = fmt::format(" {:>5} / {:>5} ", fmt::format("{:%M:%S}", elapsed), fmt::format("{:%M:%S}", seconds(0)));
    } else {
      time_info = fmt::format(" {:>5} / {:>5} ", fmt::format("{:%M:%S}", elapsed), fmt::format("{:%M:%S}", remaining));
    }
    std::string styled_time = fmt::format(time_style_, "{}", time_info);

    fmt::print("\r{}{} {} {}", styled_label, styled_bar, styled_percent, styled_time);

    if (current > total_ - 1) { fmt::print("\n"); }
  }

 private:
  std::string label_;
  int total_;
  int bar_width_;
  std::chrono::steady_clock::time_point start_time_;
  fmt::text_style label_style_;
  fmt::text_style bar_style_;
  fmt::text_style percent_style_;
  fmt::text_style time_style_;
};
}  // namespace mllm
