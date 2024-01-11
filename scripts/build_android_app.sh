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
-DNNAPI=OFF \
-DDEBUG=ON \
-DTEST=OFF \
-DARM=ON \
-DAPK=ON 

make mllm_lib -j4

# 2. copy libs
cp ./libmllm_lib.a ../android/app/src/main/cpp/libs/

# 3. build android apk
cd ../android || exit
./gradlew assembleDebug