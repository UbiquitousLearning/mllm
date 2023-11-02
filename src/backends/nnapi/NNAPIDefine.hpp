#ifndef MLLM_NNAPIDEFINE_H
#define MLLM_NNAPIDEFINE_H

#ifdef NNAPI_ENABLED
#ifdef __ANDROID__
#include "NNAPINeuralNetworks.h"
#define ANDROID_API_LEVEL (android_get_device_api_level())
#else
#undef NNAPI_ENABLED
#define NNAPI_ENABLED 0
#define ANDROID_API_LEVEL (0)
#endif
#endif

#endif // MLLM_NNAPIDEFINE_H