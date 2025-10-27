#!/bin/bash

adb shell mkdir -p /data/local/tmp/mllm/vocab
adb shell mkdir -p /data/local/tmp/mllm/qnn-lib

adb push ../vocab/qwen_vocab.mllm /data/local/tmp/mllm/vocab/


if ! adb shell [ -f "/data/local/tmp/mllm/models/showui-w8-fpbias-noshadow-xdl-test.mllm" ]; then
    adb push ../models/showui-w8-fpbias-noshadow-xdl-test.mllm "/data/local/tmp/mllm/models/showui-w8-fpbias-noshadow-xdl-test.mllm"
else
    echo "showui-w8-fpbias-noshadow-xdl-test file already exists"
fi


if ! adb shell [ -f "/data/local/tmp/mllm/models/showui-2B-rotated-q40.mllm" ]; then
    adb push ../models/showui-2B-rotated-q40.mllm "/data/local/tmp/mllm/models/showui-2B-rotated-q40.mllm"
else
    echo "showui-2B-rotated-q40.mllm file already exists"
fi

if [ -z "$QNN_SDK_ROOT" ]; then
    export QNN_SDK_ROOT=/root/research/dev/mllm/mllm/backends/qnn/sdk
    # export HEXAGON_SDK_ROOT=/root/research/dev/mllm/mllm/backends/qnn/HexagonSDK/5.4.0
    echo "QNN_SDK_ROOT is set to $QNN_SDK_ROOT"
    # exit 1
else 
    echo "QNN_SDK_ROOT is set to $QNN_SDK_ROOT"
fi

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
adb push $OP_PATH/aarch64-android/libQnnLLaMAPackage.so $DEST/libQnnLLaMAPackage_CPU.so
adb push $OP_PATH/hexagon-v75/libQnnLLaMAPackage.so $DEST/libQnnLLaMAPackage_HTP.so


if [ $? -ne 0 ]; then
    echo "adb push failed"
    exit 1
fi
# adb shell "rm /data/local/tmp/mllm/bin/qnn_context.bin"
adb push ../bin-arm-qnn/demo_showui_npu /data/local/tmp/mllm/bin/
adb shell "cd /data/local/tmp/mllm/bin && export LD_LIBRARY_PATH=/data/local/tmp/mllm/qnn-lib && export ADSP_LIBRARY_PATH=/data/local/tmp/mllm/qnn-lib && ./demo_showui_npu"