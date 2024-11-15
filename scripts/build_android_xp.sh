#!/bin/bash
mkdir ../build-arm-xp
cd ../build-arm-xp || exit

cmake .. \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_ABI="arm64-v8a" \
-DANDROID_STL=c++_static \
-DANDROID_NATIVE_API_LEVEL=android-28  \
-DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. $1 $2 $3 \
-DQNN=OFF \
-DDEBUG=OFF \
-DTEST=OFF \
-DQUANT=OFF \
-DMLLM_BUILD_XNNPACK_BACKEND=ON

make -j$(nproc)
