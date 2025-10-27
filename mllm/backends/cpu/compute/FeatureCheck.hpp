
#if defined(__linux__)
#include <sys/auxv.h>
// #if defined(__aarch64__) && !defined(HWCAP_I8MM)
#include <asm/hwcap.h> // 确保定义 HWCAP_I8MM
// #endif
#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#include <cstdio>
#include <cctype>
#include <cstring>
#include <cstdint>

#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
// 读取 ID_AA64ISAR1_EL1 寄存器
static inline uint64_t read_id_aa64isar1_el1() {
    uint64_t value;
    asm volatile("mrs %0, ID_AA64ISAR1_EL1" : "=r"(value));
    return value;
}
#endif

static bool arm_is_i8mm_supported() {
    // std::cout << "Starting i8mm detection..." << std::endl;

    // 1. macOS 专用检测（不受影响）
#if defined(__APPLE__) && defined(__aarch64__)
    // std::cout << "Using macOS sysctl detection" << std::endl;
    int supported = 0;
    size_t size = sizeof(supported);
    const int result = sysctlbyname("hw.optional.arm.FEAT_I8MM",
                                    &supported, &size, NULL, 0);
    if (result == 0 && supported) {
        // std::cout << "sysctl detection: I8MM supported!" << std::endl;
        return true;
    }
    // std::cerr << "sysctl detection "
    //           << (result ? "failed" : "I8MM not supported") << std::endl;
#endif

    // 2. 优先使用 CPU 寄存器检测（ARM64通用）
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    // std::cout << "Using ARM64 register detection method" << std::endl;
    const uint64_t isar1 = read_id_aa64isar1_el1();
    const uint8_t i8mm_field = (isar1 >> 52) & 0xF; // 位52-55
    // std::cout << "ID_AA64ISAR1_EL1 = 0x" << std::hex << isar1 << std::dec << ", I8MM field = " << static_cast<int>(i8mm_field) << std::endl;
    // 值1或2表示支持i8mm
    if (i8mm_field == 1 || i8mm_field == 2) {
        // std::cout << "Register detection: I8MM supported!" << std::endl;
        return true;
    }
    // std::cout << "Register detection: I8MM not supported" << std::endl;
#endif

// 3. /proc/cpuinfo 后备检测
#if defined(__linux__)
    // std::cout << "Using /proc/cpuinfo detection" << std::endl;
    FILE *cpuinfo = fopen("/proc/cpuinfo", "r");
    if (cpuinfo) {
        char line[512];
        while (fgets(line, sizeof(line), cpuinfo)) {
            // 检查包含"Features"的行
            if (strstr(line, "Features") && strstr(line, "i8mm")) {
                fclose(cpuinfo);
                // std::cout << "CPUinfo detection: I8MM supported!" << std::endl;
                return true;
            }
        }
        fclose(cpuinfo);
    }
    // std::cout << "CPUinfo detection: I8MM not found or file access failed" << std::endl;
#endif
    // std::cout << "No I8MM support detected" << std::endl;
    return false;
}
