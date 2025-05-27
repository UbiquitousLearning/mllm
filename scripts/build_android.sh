#!/bin/bash
# rm -rf ../build-arm
mkdir ../build-arm
cd ../build-arm || exit

cmake .. \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_ABI="arm64-v8a" \
-DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. $1 $2 $3 \
-DANDROID_PLATFORM=android-34 \
-DCMAKE_CXX_FLAGS="-march=armv8.6-a+dotprod+i8mm" \
-DDEBUG=OFF \
-DTEST=OFF \
-DARM=ON \
-DAPK=OFF

make -j$(nproc)