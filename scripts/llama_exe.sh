#!/bin/bash
echo "You need to push models to phone first: /data/local/tmp/mllm/models"

adb mkdir /data/local/tmp/mllm
adb mkdir /data/local/tmp/mllm/bin
adb mkdir /data/local/tmp/mllm/models
adb push ./libomp.so /data/local/tmp/mllm/bin
adb push ../bin-arm/main_llama /data/local/tmp/mllm/bin
adb push ../vocab/vocab.mllm /data/local/tmp/mllm/bin
adb push ../models/llama-2-7b-chat-q4_k.mllm /data/local/tmp/mllm/models
# if push failed, exit
if [ $? -ne 0 ]; then
    echo "adb push failed"
    exit 1
fi
adb shell export LD_LIBRARY_PATH=/data/local/tmp/mllm/bin
adb shell ./data/local/tmp/mllm/main_llama