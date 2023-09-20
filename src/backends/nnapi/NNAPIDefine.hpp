#ifndef NNAPIDefine_h
#define NNAPIDefine_h

#ifdef MNN_NNAPI_ENABLED
#ifdef __ANDROID__
#include "NNAPINeuralNetworks.h"
#define ANDROID_API_LEVEL (android_get_device_api_level())
#else
#undef MNN_NNAPI_ENABLED
#define MNN_NNAPI_ENABLED 0
#define ANDROID_API_LEVEL (0)
#endif
#endif

#endif /* NNAPIDefine_h */