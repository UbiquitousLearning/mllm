#ifndef NNAPIDefine_h
#define NNAPIDefine_h

// for development
#include "NNAPINeuralNetworks.h"
#define ANDROID_API_LEVEL (android_get_device_api_level())

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

#endif /* NNAPIDefine_h */