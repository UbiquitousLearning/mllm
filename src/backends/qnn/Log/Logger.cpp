//=============================================================================
//
//  Copyright (c) 2020-2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include <chrono>
#include <cstdio>
#include <iostream>
#include <sstream>

#include "LogUtils.hpp"
#include "Logger.hpp"

using namespace qnn::log;

std::shared_ptr<Logger> Logger::s_logger = nullptr;

std::mutex Logger::s_logMutex;

std::shared_ptr<Logger> Logger::createLogger(QnnLog_Callback_t callback,
                                             QnnLog_Level_t maxLevel,
                                             QnnLog_Error_t* status) {
  std::lock_guard<std::mutex> lock(s_logMutex);
  if ((maxLevel > QNN_LOG_LEVEL_VERBOSE) || (maxLevel == 0)) {
    if (status) {
      *status = QNN_LOG_ERROR_INVALID_ARGUMENT;
    }
    return nullptr;
  }
  if (!s_logger) {
    s_logger = std::shared_ptr<Logger>(new (std::nothrow) Logger(callback, maxLevel, status));
  }
  *status = QNN_LOG_NO_ERROR;
  return s_logger;
}

Logger::Logger(QnnLog_Callback_t callback, QnnLog_Level_t maxLevel, QnnLog_Error_t* status)
    : m_callback(callback), m_maxLevel(maxLevel), m_epoch(getTimestamp()) {
  if (!callback) {
    m_callback = utils::logStdoutCallback;
  }
}

void Logger::log(QnnLog_Level_t level, const char* file, long line, const char* fmt, ...) {
  if (m_callback) {
    if (level > m_maxLevel.load(std::memory_order_seq_cst)) {
      return;
    }
    va_list argp;
    va_start(argp, fmt);
    std::string logString(fmt);
    std::ignore = file;
    std::ignore = line;
    (*m_callback)(logString.c_str(), level, getTimestamp() - m_epoch, argp);
    va_end(argp);
  }
}

uint64_t Logger::getTimestamp() const {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::shared_ptr<::qnn::log::Logger> g_logger{nullptr};

bool qnn::log::initializeLogging() {
  QnnLog_Level_t logLevel;
  QnnLog_Error_t status;
#ifdef QNN_ENABLE_DEBUG
  logLevel = QNN_LOG_LEVEL_DEBUG;
#else
  logLevel = QNN_LOG_LEVEL_INFO;
#endif
  // Default log stream is enabled in Core/Logger component
  g_logger = ::qnn::log::Logger::createLogger(nullptr, logLevel, &status);
  if (QNN_LOG_NO_ERROR != status || !g_logger) {
    return false;
  }
  return true;
}

QnnLog_Callback_t qnn::log::getLogCallback() { return g_logger->getLogCallback(); }

QnnLog_Level_t qnn::log::getLogLevel() { return g_logger->getMaxLevel(); }

bool qnn::log::isLogInitialized() {
  if (g_logger == nullptr) {
    return false;
  }
  return true;
}

bool qnn::log::setLogLevel(QnnLog_Level_t maxLevel) {
  if (!::qnn::log::Logger::isValid() ||
      !(maxLevel >= QNN_LOG_LEVEL_ERROR && maxLevel <= QNN_LOG_LEVEL_DEBUG)) {
    return false;
  }

  g_logger->setMaxLevel(maxLevel);
  return true;
}
