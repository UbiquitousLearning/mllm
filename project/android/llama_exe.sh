#!/bin/bash
echo "You need to push models to phone first"
adb mkdir ./data/local/tmp/mllm
# NOTE: just for dev convinience
adb push ../bin-arm/main_llama /data/local/tmp/mllm
adb push ./vocab.mllm /data/local/tmp/mllm
# if push failed, exit
if [ $? -ne 0 ]; then
    echo "adb push failed"
    exit 1
fi
adb shell ./data/local/tmp/mllm/main_llama