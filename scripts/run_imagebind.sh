#!/bin/bash

adb shell mkdir /data/local/tmp/mllm
adb shell mkdir /data/local/tmp/mllm/bin
adb shell mkdir /data/local/tmp/mllm/models
adb shell mkdir /data/local/tmp/mllm/vocab
adb push ../vocab/clip_vocab.mllm /data/local/tmp/mllm/vocab/
adb push ../vocab/clip_merges.txt /data/local/tmp/mllm/vocab/
#adb push ../bin-arm/main_imagebind /data/local/tmp/mllm/bin/
adb push ../bin-arm/demo_imagebind /data/local/tmp/mllm/bin/
adb push ../models/imagebind_huge-q4_k.mllm /data/local/tmp/mllm/models/
# if push failed, exit
if [ $? -ne 0 ]; then
    echo "adb push failed"
    exit 1
fi
#adb shell "cd /data/local/tmp/mllm/bin && ./main_imagebind"
adb shell "cd /data/local/tmp/mllm/bin && ./demo_imagebind"