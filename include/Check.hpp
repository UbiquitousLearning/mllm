#ifndef MLLM_CHECK_H
#define MLLM_CHECK_H

#include <iostream>
#define likely(x) __builtin_expect(!!(x), 1) //x很可能为真       
#define unlikely(x) __builtin_expect(!!(x), 0) //x很可能为假
// #define COMPACT_GOOGLE_LOG_FATAL google::LogMessageFatal( \
//       __FILE__, __LINE__)


#define PREDICT(x)  !!(x)

#define LOG_IF(condition) \
  static_cast<void>(0),             \
  !(condition) ? (void) 0 : COMPACT_GOOGLE_LOG_FATAL

#define CHECK(condition)  \
      if(!PREDICT(condition)) \
            std::cout << "Check failed: " #condition " "<<std::endl



// void CHECK(condition)  \
//     //   LOG_IF(GOOGLE_PREDICT_BRANCH_NOT_TAKEN(!(condition))) \
//              << "Check failed: " #condition " "
//     if(unlikely(x))
//         std::cout<< << "Check failed: " #condition " "<<std::endl;
    

#define CHECK_OP(name, op, val1, val2) CHECK((val1) op (val2))

#define CHECK_EQ(val1, val2) CHECK_OP(_EQ, ==, val1, val2)
#define CHECK_NE(val1, val2) CHECK_OP(_NE, !=, val1, val2)
#define CHECK_LE(val1, val2) CHECK_OP(_LE, <=, val1, val2)
#define CHECK_LT(val1, val2) CHECK_OP(_LT, < , val1, val2)
#define CHECK_GE(val1, val2) CHECK_OP(_GE, >=, val1, val2)
#define CHECK_GT(val1, val2) CHECK_OP(_GT, > , val1, val2)


#endif //MLLM_CHECK_H