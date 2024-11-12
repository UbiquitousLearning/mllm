#ifndef MLLM_LOGGER_HPP_
#define MLLM_LOGGER_HPP_

#include "fmt/format.h" //< fmt / format.h>
#include <string_view>
#include <chrono>
#include <sstream>
#include <iostream>
#ifdef ANDROID_API
#include <android/log.h>
#endif

// Log levels
enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};
// 编译期字符串
template <std::size_t N>
struct ConstString {
    char data[N];
    constexpr ConstString(const char (&str)[N]) {
        for (std::size_t i = 0; i < N; ++i) {
            data[i] = str[i];
        }
    }

    constexpr char operator[](std::size_t i) const {
        return data[i];
    }

    constexpr std::size_t size() const {
        return N - 1;
    }

    constexpr operator std::string_view() const {
        return std::string_view(data, N - 1);
    }
};
template <std::size_t N>
struct StringBuilder {
    char data[N];
    std::size_t len;

    constexpr StringBuilder() :
        data{}, len(0) {
    }

    constexpr void append(char c) {
        if (len < N - 1) {
            data[len++] = c;
        }
    }

    constexpr void append(const char *s, std::size_t slen) {
        for (std::size_t i = 0; i < slen && len < N - 1; ++i) {
            data[len++] = s[i];
        }
    }

    constexpr ConstString<N> build() const {
        char result[N]{};
        for (std::size_t i = 0; i < len; ++i) {
            result[i] = data[i];
        }
        return ConstString<N>(result);
    }
};

template <typename T, T... chars>
struct FormatStringHelper;
template <std::size_t N>
constexpr auto convert_format(const char (&fmt)[N]) {
    StringBuilder<N * 2> builder;

    for (std::size_t i = 0; i < N - 1; ++i) { // N-1 忽略结尾的null
        if (fmt[i] == '%' && i + 1 < N - 1) {
            switch (fmt[i + 1]) {
            case 's':
                builder.append('{');
                builder.append('}');
                ++i; // 跳过下一个字符
                break;
            case 'd':
                builder.append('{');
                builder.append(':');
                builder.append('d');
                builder.append('}');
                ++i;
                break;
            case 'f':
                builder.append('{');
                builder.append(':');
                builder.append('f');
                builder.append('}');
                ++i;
                break;
            case '%':
                builder.append('%');
                ++i;
                break;
            default:
                builder.append(fmt[i]);
            }
        } else {
            builder.append(fmt[i]);
        }
    }

    return builder.build();
}

template <typename T, T... chars>
constexpr auto operator""_fmt() {
    constexpr char str[sizeof...(chars) + 1] = {chars..., '\0'};
    return convert_format(str);
}
template <typename... Args>
constexpr auto format_legacy(std::string_view format, Args &&...args) {
    return fmt::format(format, std::forward<Args>(args)...);
}
#define FORMAT_STR_LEGACY(str) []() {         \
    constexpr auto fmt = convert_format(str); \
    return fmt;                               \
}()

#define FORMAT_LEGACY(fmt, ...) format_legacy(FORMAT_STR_LEGACY(fmt), ##__VA_ARGS__)

class Logger {
public:
    static Logger &getInstance() {
        static Logger instance;
        return instance;
    }

    template <typename... Args>
    void log(LogLevel level,
             const char *file,
             int line,
             std::string_view format,
             Args &&...args) {
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::system_clock::to_time_t(now);

        std::string levelStr = getLevelString(level);
        std::string formatted_msg = fmt::format(format, std::forward<Args>(args)...);

#ifdef ANDROID_API
        std::string final_message = fmt::format("[{}:{}] {}",
                                                file,
                                                line,
                                                formatted_msg);
        android_LogPriority priority = getAndroidPriority(level);
        __android_log_print(priority, "MLLM", "%s", final_message.c_str());
#else
        std::string final_message = fmt::format("[{}] {} [{}:{}] {}",
                                                levelStr,
                                                std::string(std::ctime(&timestamp)).substr(0, 24),
                                                file,
                                                line,
                                                formatted_msg);
        std::cout << final_message << std::endl;
#endif
    }

    // Convenience methods for different log levels
    template <typename... Args>
    void debug(const char *file,
               int line,
               std::string_view format,
               Args &&...args) {
        log(LogLevel::DEBUG, file, line, format, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void info(const char *file,
              int line,
              std::string_view format,
              Args &&...args) {
        log(LogLevel::INFO, file, line, format, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void warning(const char *file,
                 int line,
                 std::string_view format,
                 Args &&...args) {
        log(LogLevel::WARNING, file, line, format, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void error(const char *file,
               int line,
               std::string_view format,
               Args &&...args) {
        log(LogLevel::ERROR, file, line, format, std::forward<Args>(args)...);
    }

private:
    Logger() = default;
    ~Logger() = default;
    Logger(const Logger &) = delete;
    Logger &operator=(const Logger &) = delete;

    static std::string getLevelString(LogLevel level) {
        switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR: return "ERROR";
        default: return "UNKNOWN";
        }
    }

#ifdef ANDROID_API
    static android_LogPriority getAndroidPriority(LogLevel level) {
        switch (level) {
        case LogLevel::DEBUG: return ANDROID_LOG_DEBUG;
        case LogLevel::INFO: return ANDROID_LOG_INFO;
        case LogLevel::WARNING: return ANDROID_LOG_WARN;
        case LogLevel::ERROR: return ANDROID_LOG_ERROR;
        default: return ANDROID_LOG_DEFAULT;
        }
    }
#endif
};
class LogStream {
public:
    LogStream(LogLevel level, const char *file, int line) :
        level_(level), file_(file), line_(line) {
    }

    ~LogStream() {
        // 在析构时输出完整的日志消息
        Logger::getInstance().log(level_, file_, line_, stream_.str());
    }

    template <typename T>
    LogStream &operator<<(const T &value) {
        stream_ << value;
        return *this;
    }

    // 支持std::endl等流控制符
    LogStream &operator<<(std::ostream &(*manip)(std::ostream &)) {
        manip(stream_);
        return *this;
    }

private:
    LogLevel level_;
    const char *file_;
    int line_;
    std::ostringstream stream_;
};

// Convenience macros for logging
#define MLLM_LOG_DEBUG(format, ...) \
    Logger::getInstance().debug(__FILE__, __LINE__, format, ##__VA_ARGS__)

#define MLLM_LOG_INFO(format, ...) \
    Logger::getInstance().info(__FILE__, __LINE__, format, ##__VA_ARGS__)

#define MLLM_LOG_WARNING(format, ...) \
    Logger::getInstance().warning(__FILE__, __LINE__, format, ##__VA_ARGS__)

#define MLLM_LOG_ERROR(format, ...) \
    Logger::getInstance().error(__FILE__, __LINE__, format, ##__VA_ARGS__)

// For compatibility with existing code
#define MLLM_LOG_STREAM(level) LogStream(level, __FILE__, __LINE__)
#define MLLM_LOG_DEBUG_STREAM MLLM_LOG_STREAM(LogLevel::DEBUG)
#define MLLM_LOG_INFO_STREAM MLLM_LOG_STREAM(LogLevel::INFO)
#define MLLM_LOG_WARNING_STREAM MLLM_LOG_STREAM(LogLevel::WARNING)
#define MLLM_LOG_ERROR_STREAM MLLM_LOG_STREAM(LogLevel::ERROR)

// For compatibility with QNN_XXXX
#define MLLM_LOG_DEBUG_LEGACY(format, ...) \
    MLLM_LOG_DEBUG(FORMAT_STR_LEGACY(format), ##__VA_ARGS__)
#define MLLM_LOG_INFO_LEGACY(format, ...) \
    MLLM_LOG_INFO(FORMAT_STR_LEGACY(format), ##__VA_ARGS__)
#define MLLM_LOG_WARN_LEGACY(format, ...) \
    MLLM_LOG_WARNING(FORMAT_STR_LEGACY(format), ##__VA_ARGS__)
#define MLLM_LOG_ERROR_LEGACY(format, ...) \
    MLLM_LOG_ERROR(FORMAT_STR_LEGACY(format), ##__VA_ARGS__)

#endif // LOGGER_HPP_