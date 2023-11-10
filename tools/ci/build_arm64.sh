#!bash
# Check if ${ANDROID_NDK} exists else set it to /opt/ndk
if [ -z "${ANDROID_NDK}" ]; then
    ANDROID_NDK=/opt/ndk
fi
mkdir -p bin
cmake -DTEST=on  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" -DANDROID_ABI="arm64-v8a" -DANDROID_NDK="${ANDROID_NDK}" -DANDROID_PLATFORM=android-28 -B./build_arm/ .
cmake --build ./build_arm/ --target all -- -j$(nproc)

(cd test && bash test.sh )
ls -l ./bin-arm/

