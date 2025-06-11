#!/bin/bash
mkdir ../build-arm-qnn
cd ../build-arm-qnn || exit

cmake .. \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_ABI="arm64-v8a" \
-DANDROID_STL=c++_static \
-DANDROID_NATIVE_API_LEVEL=android-28  \
-DCMAKE_CXX_FLAGS="-march=armv8.2-a+dotprod" \
-DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. $1 $2 $3 \
-DQNN=ON \
-DDEBUG=OFF \
-DTEST=OFF \
-DQUANT=OFF \
-DQNN_VALIDATE_NODE=ON \
-DMLLM_BUILD_XNNPACK_BACKEND=OFF

make -j$(nproc)
