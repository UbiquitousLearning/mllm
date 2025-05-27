#!/bin/bash

adb shell mkdir /data/local/tmp/mllm
adb shell mkdir /data/local/tmp/mllm/bin
adb shell mkdir /data/local/tmp/mllm/models
adb shell mkdir /data/local/tmp/mllm/vocab
adb push ../vocab/* /data/local/tmp/mllm/vocab/
adb push ../assets/* /data/local/tmp/mllm/assets/
adb push ../bin-arm/demo_qwen2_vl /data/local/tmp/mllm/bin/
adb push ../models/qwen-2-vl-2b-instruct-q4_k.mllm /data/local/tmp/mllm/models/
# if push failed, exit
if [ $? -ne 0 ]; then
    echo "adb push failed"
    exit 1
fi
adb shell "cd /data/local/tmp/mllm/bin && ./demo_qwen2_vl"