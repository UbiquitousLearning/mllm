!/bin/bash

adb shell mkdir -p /data/local/tmp/mllm/vocab
adb shell mkdir -p /data/local/tmp/mllm/qnn-lib

adb push ../vocab/phonelm_vocab.mllm /data/local/tmp/mllm/vocab/


if ! adb shell [ -f "/data/local/tmp/mllm/models/phonelm-1.5b-instruct-int8.mllm" ]; then
    adb push ../models/phonelm-1.5b-instruct-int8.mllm "/data/local/tmp/mllm/models/phonelm-1.5b-instruct-int8.mllm"
else
    echo "phonelm-1.5b-instruct-int8 file already exists"
fi


if ! adb shell [ -f "/data/local/tmp/mllm/models//phonelm-1.5b-instruct-q4_0_4_4.mllm" ]; then
    adb push ../models//phonelm-1.5b-instruct-q4_0_4_4.mllm "/data/local/tmp/mllm/models//phonelm-1.5b-instruct-q4_0_4_4.mllm"
else
    echo "/phonelm-1.5b-instruct-q4_0_4_4.mllm file already exists"
fi

if [ -z "$QNN_SDK_ROOT" ]; then
    export QNN_SDK_ROOT=/root/research/dev/mllm/src/backends/qnn/sdk
    echo "QNN_SDK_ROOT is set to $QNN_SDK_ROOT"
else 
    echo "QNN_SDK_ROOT is set to $QNN_SDK_ROOT"
fi

ANDR_LIB=$QNN_SDK_ROOT/lib/aarch64-android
OP_PATH=../src/backends/qnn/LLaMAOpPackageHtp/LLaMAPackage/build
DEST=/data/local/tmp/mllm/qnn-lib

adb push $ANDR_LIB/libQnnHtp.so $DEST
adb push $ANDR_LIB/libQnnHtpV75Stub.so $DEST
adb push $ANDR_LIB/libQnnHtpPrepare.so $DEST
adb push $ANDR_LIB/libQnnHtpProfilingReader.so $DEST
adb push $ANDR_LIB/libQnnHtpOptraceProfilingReader.so $DEST
adb push $ANDR_LIB/libQnnHtpV75CalculatorStub.so $DEST
adb push $LIBPATH/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so $DEST
adb push $OP_PATH/aarch64-android/libQnnLLaMAPackage.so $DEST/libQnnLLaMAPackage_CPU.so
adb push $OP_PATH/hexagon-v75/libQnnLLaMAPackage.so $DEST/libQnnLLaMAPackage_HTP.so


if [ $? -ne 0 ]; then
    echo "adb push failed"
    exit 1
fi

adb push ../bin-arm/demo_phonelm_npu /data/local/tmp/mllm/bin/
adb shell "cd /data/local/tmp/mllm/bin && export LD_LIBRARY_PATH=/data/local/tmp/mllm/qnn-lib && export ADSP_LIBRARY_PATH=/data/local/tmp/mllm/qnn-lib && ./demo_phonelm_npu"