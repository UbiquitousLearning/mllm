adb shell mkdir -p /data/local/tmp/mllm/qnn-lib

ANDR_LIB=$QNN_SDK_ROOT/lib/aarch64-android
OP_PATH=../mllm/backends/qnn/LLaMAOpPackageHtp/LLaMAPackage/build
DEST=/data/local/tmp/mllm/qnn-lib

adb push $ANDR_LIB/libQnnHtp.so $DEST
adb push $ANDR_LIB/libQnnHtpV75Stub.so $DEST
adb push $ANDR_LIB/libQnnHtpPrepare.so $DEST
adb push $ANDR_LIB/libQnnHtpProfilingReader.so $DEST
adb push $ANDR_LIB/libQnnHtpOptraceProfilingReader.so $DEST
adb push $ANDR_LIB/libQnnHtpV75CalculatorStub.so $DEST
adb push $QNN_SDK_ROOT/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so $DEST
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so $DEST

adb push $OP_PATH/aarch64-android/libQnnLLaMAPackage.so $DEST/libQnnLLaMAPackage_CPU.so
adb push $OP_PATH/hexagon-v75/libQnnLLaMAPackage.so $DEST/libQnnLLaMAPackage_HTP.so
