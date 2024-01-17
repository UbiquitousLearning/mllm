#!/bin/bash
#echo "You need to push models to phone first: /data/local/tmp/mllm/models"

adb mkdir /data/local/tmp/mllm
adb mkdir /data/local/tmp/mllm/bin
adb mkdir /data/local/tmp/mllm/models
adb mkdir /data/local/tmp/mllm/vocab
adb push ../vocab/llama_vocab.mllm /data/local/tmp/mllm/vocab
adb push ../bin-arm/main_llama /data/local/tmp/mllm/bin
adb push ../vocab/vocab.mllm /data/local/tmp/mllm/bin
adb push ../models/llama-2-7b-chat-q4_k.mllm /data/local/tmp/mllm/models
# if push failed, exit
if [ $? -ne 0 ]; then
    echo "adb push failed"
    exit 1
fi
adb shell cd /data/local/tmp/mllm/bin&&./main_llama