#!/bin/bash
mkdir ../build-arm
cd ../build-arm || exit

# 1. build mllm_lib
cmake .. \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_ABI="arm64-v8a" \
-DANDROID_NATIVE_API_LEVEL=android-28  \
-DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. $1 $2 $3 \
-DARM=ON \
-DAPK=ON \
-DQNN=ON \
-DDEBUG=OFF \
-DTEST=OFF \
-DQUANT=OFF \
-DQNN_VALIDATE_NODE=ON \
-DMLLM_BUILD_XNNPACK_BACKEND=OFF


make mllm_lib -j16

# # 2. copy libs
# cp ./libmllm_lib.a ../android/app/src/main/cpp/libs/

# # 3. build android apk
# cd ../android || exit
# ./gradlew assembleDebug