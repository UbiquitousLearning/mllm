# Qualcomm AI Engine Direct(QNN) backend

## QNN Environment Set Up

## Op Package Compile

To use QNN offload, the CPU & HTP QNN op package are needed, the following scripts will build QNN op package needed by the project.

```bash
export PATH=$PATH:/mllm/platform-tools
export QNN_SDK_ROOT=/mllm/qualcomm_ai_engine_direct_220/
export ANDROID_NDK=/mllm/android-ndk-r26d
export ANDROID_NDK_ROOT=/mllm/android-ndk-r26d
export PATH=$PATH:$ANDROID_NDK_ROOT

source /mllm/HexagonSDK/setup_sdk_env.source
source $QNN_SDK_ROOT/bin/envsetup.sh

make htp_aarch64 && make htp_v75
```

Push QNN libraris and QNN op packages to device:

```bash
LIBPATH=$qualcomm_ai_engine_direct_220
ANDR_LIB=$LIBPATH/lib/aarch64-android
DEST=/data/local/tmp/mllm/qnn-lib

adb push $ANDR_LIB/libQnnHtp.so $DEST
adb push $ANDR_LIB/libQnnHtpV75Stub.so $DEST
adb push $ANDR_LIB/libQnnHtpPrepare.so $DEST
adb push $ANDR_LIB/libQnnHtpProfilingReader.so $DEST
adb push $ANDR_LIB/libQnnHtpOptraceProfilingReader.so $DEST
adb push $ANDR_LIB/libQnnHtpV75CalculatorStub.so $DEST
adb push $LIBPATH/libQnnHtpV75Skel.so $DEST

OP_PATH=LLaMAOpPackageHtp/LLaMAPackage/build

DEST=/data/local/tmp/mllm/qnn-lib
adb -s 10.29.208.59:9808 push $OP_PATH/aarch64-android/libQnnLLaMAPackage.so $DEST/libQnnLLaMAPackage_CPU.so
adb -s 10.29.208.59:9808 push $OP_PATH/hexagon-v75/libQnnLLaMAPackage.so $DEST/libQnnLLaMAPackage_HTP.so
```