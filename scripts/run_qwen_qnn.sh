#!/bin/bash

adb shell mkdir -p /data/local/tmp/mllm/vocab
adb shell mkdir -p /data/local/tmp/mllm/qnn-lib

adb push ../vocab/qwen_vocab.mllm /data/local/tmp/mllm/vocab/


if ! adb shell [ -f "/data/local/tmp/mllm/models/qwen-1.5-1.8b-chat-int8.mllm" ]; then
    adb push ../models/qwen-1.5-1.8b-chat-int8.mllm "/data/local/tmp/mllm/models/qwen-1.5-1.8b-chat-int8.mllm"
else
    echo "qwen-1.5-1.8b-chat-int8 file already exists"
fi


if ! adb shell [ -f "/data/local/tmp/mllm/models/qwen-1.5-1.8b-chat-q4k.mllm" ]; then
    adb push ../models/qwen-1.5-1.8b-chat-q4k.mllm "/data/local/tmp/mllm/models/qwen-1.5-1.8b-chat-q4k.mllm"
else
    echo "qwen-1.5-1.8b-chat-q4k.mllm file already exists"
fi

LIBPATH=../src/backends/qnn/qualcomm_ai_engine_direct_220/
ANDR_LIB=$LIBPATH/lib/aarch64-android
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

adb push ../bin-arm/demo_qwen_npu /data/local/tmp/mllm/bin/
adb shell "cd /data/local/tmp/mllm/bin && export LD_LIBRARY_PATH=/data/local/tmp/mllm/qnn-lib && export ADSP_LIBRARY_PATH=/data/local/tmp/mllm/qnn-lib && ./demo_qwen_npu"