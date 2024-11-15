#!/bin/bash
mkdir ../build-arm
cd ../build-arm || exit

cmake .. \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_ABI="arm64-v8a" \
-DANDROID_NATIVE_API_LEVEL=android-28  \
-DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. $1 $2 $3 \
-DDEBUG=OFF \
-DTEST=OFF \
-DARM=ON \
-DAPK=OFF

make -j$(nproc)
