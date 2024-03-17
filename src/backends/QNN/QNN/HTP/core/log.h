//==============================================================================
//
// Copyright (c) 2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef LOG_H
#define LOG_H 1

#include "weak_linkage.h"
#include "macros_attribute.h"
#include <cstdarg>
#include <string>
#include <chrono>

#if !defined(__PRETTY_FUNCTION__) && !defined(__GNUC__)
#define __FUNC_INFO__ __FUNCSIG__
#else
#define __FUNC_INFO__ __PRETTY_FUNCTION__
#endif

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X)        STRINGIZE_DETAIL(X)

// Constexpr that will strip the path off of the file for logging purposes
#ifdef __cplusplus
constexpr
#endif
        char const *
        stripFilePath(const char *path)
{
    const char *file = path;
    while (*path) {
        if (*path++ == '/') {
            file = path;
        }
    }
    return file;
}

#include "graph_status.h"
#include "cc_pp.h"

#ifdef __cplusplus
#include <cstdio>
#else
#include <stdio.h>
#endif

// If log level or the dynamic logging flag are defined but don't have a value,
// then consider them to be undefined.
#if ~(~NN_LOG_MAXLVL + 0) == 0 && ~(~NN_LOG_MAXLVL + 1) == 1
#undef NN_LOG_MAXLVL
#endif

#if ~(~NN_LOG_DYNLVL + 0) == 0 && ~(~NN_LOG_DYNLVL + 1) == 1
#undef NN_LOG_DYNLVL
#endif

/*
 * We have migrated using C++ features like iostream to printf strings.
 * Why?
 * * C++ iostream makes it more difficult to use mixed decimal/hex
 * * C++ iostream isn't easily compatible with on-target logging facilities
 * * C++ iostream is bad for code size, printf is much better
 */

//Log levels macro
#define NN_LOG_ERRORLVL         0 //Error log level is 0
#define NN_LOG_WARNLVL          1 //Warning log level is 1
#define NN_LOG_STATLVL          2 //Stats log level is 2
#define NN_LOG_INFOLVL          3 //Info log level is 3
#define NN_LOG_VERBOSELVL       4 //Verbose log level is from 4-10
#define NN_LOG_STATLVL_INTERNAL 8
#define NN_LOG_INFOLVL_INTERNAL 9
#define NN_LOG_DEBUGLVL         11 //Debug log level is > 10

typedef void (*DspLogCallbackFunc)(int level, const char *fmt, va_list args);

// Dynamically set the logging priority level.
PUSH_VISIBILITY(default)
EXTERN_C_BEGIN
extern "C" {
API_FUNC_EXPORT void SetLogPriorityLevel(int level);
API_FUNC_EXPORT int GetLogPriorityLevel();
API_FUNC_EXPORT void SetLogCallbackFunc(DspLogCallbackFunc fn);
API_FUNC_EXPORT DspLogCallbackFunc GetLogCallbackFunc();
}
EXTERN_C_END
POP_VISIBILITY()

#ifdef __cplusplus
extern "C" {
#endif

// special log message for x86 that will log regardless logging level
void qnndsp_x86_log(const char *fmt, ...);

#ifdef __cplusplus
}
#endif

/////////////////////////ENABLE_QNN_LOG
#ifdef ENABLE_QNNDSP_LOG

PUSH_VISIBILITY(default)
#include "weak_linkage.h"

API_FUNC_EXPORT API_C_FUNC void API_FUNC_NAME(SetLogCallback)(DspLogCallbackFunc cbFn, int logPriority);

extern "C" {
API_FUNC_EXPORT void qnndsp_log(int prio, const char *FMT, ...);

API_FUNC_EXPORT void hv3_load_log_functions(decltype(SetLogCallback) **SetLogCallback_f);
}
POP_VISIBILITY()

#define qnndsp_base_log(prio, cformat, ...) (void)(qnndsp_log(prio, cformat, ##__VA_ARGS__))

#define rawlog(cformat, ...) (qnndsp_base_log(NN_LOG_ERRORLVL, cformat, ##__VA_ARGS__), GraphStatus::ErrorFatal)
#define okaylog(cformat, ...)                                                                                          \
    (qnndsp_base_log(NN_LOG_ERRORLVL, "%s:" STRINGIZE(__LINE__) ":" cformat "\n", stripFilePath(__FILE__),             \
                     ##__VA_ARGS__),                                                                                   \
     GraphStatus::ErrorFatal)
#define errlog(cformat, ...)                                                                                           \
    (qnndsp_base_log(NN_LOG_ERRORLVL, "%s:" STRINGIZE(__LINE__) ":ERROR:" cformat "\n", stripFilePath(__FILE__),       \
                     ##__VA_ARGS__),                                                                                   \
     GraphStatus::ErrorFatal)
#define warnlog(cformat, ...)        qnndsp_base_log(NN_LOG_WARNLVL, "WARNING: " cformat "\n", ##__VA_ARGS__)
#define statlog(statname, statvalue) qnndsp_base_log(NN_LOG_STATLVL, "STAT: %s=%lld\n", statname, (long long)statvalue)
#define i_statlog(statname, statvalue)                                                                                 \
    qnndsp_base_log(NN_LOG_STATLVL_INTERNAL, "STAT: %s=%lld\n", statname, (long long)statvalue)
#define statslog(statname, statvalue)   qnndsp_base_log(NN_LOG_STATLVL, "STAT: %s=%s\n", statname, statvalue)
#define i_statslog(statname, statvalue) qnndsp_base_log(NN_LOG_STATLVL_INTERNAL, "STAT: %s=%s\n", statname, statvalue)
#define infolog(cformat, ...)           qnndsp_base_log(NN_LOG_INFOLVL, cformat "\n", ##__VA_ARGS__)
#define i_infolog(cformat, ...)         qnndsp_base_log(NN_LOG_INFOLVL_INTERNAL, cformat "\n", ##__VA_ARGS__)
#define _debuglog(cformat, ...)         qnndsp_base_log(NN_LOG_DEBUGLVL, cformat "\n", ##__VA_ARGS__)
#define verboselog(cformat, ...)        qnndsp_base_log(NN_LOG_VERBOSELVL, cformat "\n", ##__VA_ARGS__)
#define logmsgraw(prio, cformat, ...)   (void)(qnndsp_base_log(prio, cformat, ##__VA_ARGS__))
#define logmsg(prio, cformat, ...)                                                                                     \
    (void)(qnndsp_base_log(prio, "%s:" STRINGIZE(__LINE__) ":" cformat "\n", stripFilePath(__FILE__), ##__VA_ARGS__))
#define logmsgl(prio, cformat, ...) (void)(qnndsp_base_log(prio, cformat, ##__VA_ARGS__))

#else //Hexagon default log
#define rawlog(FMT, ...) (printf(FMT, ##__VA_ARGS__), fflush(stdout), GraphStatus::ErrorFatal)
#define okaylog(FMT, ...)                                                                                              \
    (printf("%s:" STRINGIZE(__LINE__) ":" FMT "\n", stripFilePath(__FILE__), ##__VA_ARGS__), fflush(stdout),           \
     GraphStatus::ErrorFatal)
#define errlog(FMT, ...)                                                                                               \
    (printf("%s:" STRINGIZE(__LINE__) ":ERROR:" FMT "\n", stripFilePath(__FILE__), ##__VA_ARGS__), fflush(stdout),     \
     GraphStatus::ErrorFatal)
#define errlogl(FMT, ...) (printf(FMT, ##__VA_ARGS__), GraphStatus::ErrorFatal)
#if defined(NN_LOG_DYNLVL) && (NN_LOG_DYNLVL > 0)
#define logmsgraw(PRIO, FMT, ...)                                                                                      \
    (void)({                                                                                                           \
        if (PRIO <= GetLogPriorityLevel()) rawlog(FMT, ##__VA_ARGS__);                                                 \
    })
#define logmsg(PRIO, FMT, ...)                                                                                         \
    (void)({                                                                                                           \
        if (PRIO <= GetLogPriorityLevel()) okaylog(FMT, ##__VA_ARGS__);                                                \
    })
#define logmsgl(PRIO, FMT, ...)                                                                                        \
    (void)({                                                                                                           \
        if (PRIO <= GetLogPriorityLevel()) errlogl(FMT, ##__VA_ARGS__);                                                \
    })
#elif defined(NN_LOG_MAXLVL)
#ifdef __cplusplus
constexpr
#endif
        static bool
        log_condition(const int prio)
{
    return ((prio <= NN_LOG_MAXLVL) ? true : false);
};
#define logmsgraw(PRIO, FMT, ...)                                                                                      \
    (void)({                                                                                                           \
        if (log_condition(PRIO)) rawlog(FMT, ##__VA_ARGS__);                                                           \
    })
#define logmsg(PRIO, FMT, ...)                                                                                         \
    (void)({                                                                                                           \
        if (log_condition(PRIO)) okaylog(FMT, ##__VA_ARGS__);                                                          \
    })
#define logmsgl(PRIO, FMT, ...)                                                                                        \
    (void)({                                                                                                           \
        if (log_condition(PRIO)) errlogl(FMT, ##__VA_ARGS__);                                                          \
    })
#else
#define logmsgraw(PRIO, FMT, ...) (void)(rawlog(FMT, ##__VA_ARGS__))
#define logmsg(PRIO, FMT, ...)    (void)(okaylog(FMT, ##__VA_ARGS__))
#define logmsgl(PRIO, FMT, ...)   (void)(errlogl(FMT, ##__VA_ARGS__))
#endif
#define warnlog(FMT, ...)               logmsg(NN_LOG_WARNLVL, "WARNING: " FMT, ##__VA_ARGS__)
#define statlog(statname, statvalue)    logmsg(NN_LOG_STATLVL, "STAT: %s=%lld", statname, (long long)statvalue)
#define i_statlog(statname, statvalue)  logmsg(NN_LOG_STATLVL_INTERNAL, "STAT: %s=%lld", statname, (long long)statvalue)
#define statslog(statname, statvalue)   logmsg(NN_LOG_STATLVL, "STAT: %s=%s", statname, statvalue)
#define i_statslog(statname, statvalue) logmsg(NN_LOG_STATLVL_INTERNAL, "STAT: %s=%s", statname, (statvalue))
#define infolog(FMT, ...)               logmsg(NN_LOG_INFOLVL, FMT, ##__VA_ARGS__)
#define i_infolog(FMT, ...)             logmsg(NN_LOG_INFOLVL_INTERNAL, FMT, ##__VA_ARGS__)
#define _debuglog(FMT, ...)             logmsg(NN_LOG_DEBUGLVL, FMT, ##__VA_ARGS__)
#define verboselog(FMT, ...)            logmsg(NN_LOG_VERBOSELVL, FMT, ##__VA_ARGS__)
#endif
#define debuglog(...) _debuglog(__VA_ARGS__)

#ifdef NN_LOG_MAXLVL
#define LOG_STAT()    ((NN_LOG_MAXLVL) >= NN_LOG_STATLVL)
#define LOG_INFO()    ((NN_LOG_MAXLVL) >= NN_LOG_INFOLVL)
#define LOG_DEBUG()   ((NN_LOG_MAXLVL) >= NN_LOG_DEBUGLVL)
#define LOG_VERBOSE() ((NN_LOG_MAXLVL) >= NN_LOG_VERBOSELVL)
#else
#define LOG_STAT()    (1)
#define LOG_INFO()    (1)
#define LOG_DEBUG()   (1)
#define LOG_VERBOSE() (1)
#endif //#ifdef NN_LOG_MAXLVL

class ExternalProgressLogger {

  public:
    static void start(const char *stage_name);

    static void update_progress(unsigned int numerator, unsigned int denominator);

    static void end(const char *stage_name, const char *duration);
};

class ExternalTimePoint {
    using TimePoint = std::chrono::high_resolution_clock::time_point;
    const std::string stage_name;
    const TimePoint start_time;
    unsigned int numerator = 1;
    unsigned int denominator = 1;
    bool done = false;

  public:
    explicit ExternalTimePoint(const std::string &&stage_name);

    void update_progress(unsigned int new_numerator, unsigned int new_denominator);

    void close();

    // Custom destructor
    ExternalTimePoint() = delete;
    ExternalTimePoint(const ExternalTimePoint &) = delete;
    ExternalTimePoint &operator=(ExternalTimePoint &t) = delete;
    ExternalTimePoint(ExternalTimePoint &&) = delete;
    ExternalTimePoint &operator=(ExternalTimePoint &&t) = delete;
    ~ExternalTimePoint() { close(); }
};
#endif //#ifndef LOG_H