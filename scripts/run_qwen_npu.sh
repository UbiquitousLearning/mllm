#!/bin/bash

adb shell mkdir /data/local/tmp/mllm
adb shell mkdir /data/local/tmp/mllm/bin
adb shell mkdir /data/local/tmp/mllm/models
adb shell mkdir /data/local/tmp/mllm/vocab
adb shell mkdir /data/local/tmp/mllm/qnn-lib

adb push ../vocab/qwen_vocab.mllm /data/local/tmp/mllm/vocab/
adb push ../bin-arm/main_qwen_npu /data/local/tmp/mllm/bin/
adb push ../models/qwen-1.5-1.8b-chat-int8.mllm /data/local/tmp/mllm/models/
adb push ../models/qwen-1.5-1.8b-chat-q4k.mllm /data/local/tmp/mllm/models/

# check if qnn env is set up
# if [ -z "$QNN_SDK_ROOT" ]; then
#     echo "QNN_SDK_ROOT is not set"
#     exit 1
# else 
#     echo "QNN_SDK_ROOT is set to $QNN_SDK_ROOT"
# fi

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

# if push failed, exit
if [ $? -ne 0 ]; then
    echo "adb push failed"
    exit 1
fi

adb shell "cd /data/local/tmp/mllm/bin && export LD_LIBRARY_PATH=/data/local/tmp/mllm/qnn-lib && export ADSP_LIBRARY_PATH=/data/local/tmp/mllm/qnn-lib && ./main_qwen_npu -s 64 -c 1"