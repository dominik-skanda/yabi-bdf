#ifndef YABI_LOG_HPP
#define YABI_LOG_HPP

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace yabi {

enum class LogLevel : int { DEBUG = 0, INFO = 1, WARNING = 2, ERROR = 3, FATAL = 4 };

class Logger {
  public:
    static void setLevel(LogLevel lvl) {
        std::lock_guard<std::mutex> lock(mutex_);
        level_ = lvl;
    }
    static LogLevel level() {
        std::lock_guard<std::mutex> lock(mutex_);
        return level_;
    }

    template <typename... Args>
    static void log(LogLevel lvl, const char *file, int line, Args &&...args) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (lvl < level_)
            return;
        std::ostringstream oss;
        oss << "[" << timeString() << "] ";
        switch (lvl) {
        case LogLevel::DEBUG:
            oss << "DEBUG";
            break;
        case LogLevel::INFO:
            oss << "INFO";
            break;
        case LogLevel::WARNING:
            oss << "WARN";
            break;
        case LogLevel::ERROR:
            oss << "ERROR";
            break;
        case LogLevel::FATAL:
            oss << "FATAL";
            break;
        }
        oss << " [" << fileName(file) << ":" << line << "] ";
        (oss << ... << std::forward<Args>(args));
        oss << std::endl;
        std::cerr << oss.str();
        std::flush(std::cerr);
    }

  private:
    static std::mutex mutex_;
    static LogLevel level_;

    static std::string timeString() {
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }
    static std::string fileName(const char *path) {
        std::string s(path);
        size_t p = s.find_last_of("/\\");
        return (p == std::string::npos) ? s : s.substr(p + 1);
    }
};

inline std::mutex Logger::mutex_;
inline LogLevel Logger::level_ = LogLevel::INFO;

} // namespace yabi

// Convenience macros (always include file/line context)
#define YABI_LOG_DEBUG(...)                                                                        \
    yabi::Logger::log(yabi::LogLevel::DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define YABI_LOG_INFO(...) yabi::Logger::log(yabi::LogLevel::INFO, __FILE__, __LINE__, __VA_ARGS__)
#define YABI_LOG_WARN(...)                                                                         \
    yabi::Logger::log(yabi::LogLevel::WARNING, __FILE__, __LINE__, __VA_ARGS__)
#define YABI_LOG_ERROR(...)                                                                        \
    yabi::Logger::log(yabi::LogLevel::ERROR, __FILE__, __LINE__, __VA_ARGS__)
#define YABI_LOG_FATAL(...)                                                                        \
    yabi::Logger::log(yabi::LogLevel::FATAL, __FILE__, __LINE__, __VA_ARGS__)

#endif // YABI_LOG_HPP
