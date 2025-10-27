#!/bin/bash

adb shell mkdir /data/local/tmp/mllm
adb shell mkdir /data/local/tmp/mllm/bin
adb shell mkdir /data/local/tmp/mllm/models
adb shell mkdir /data/local/tmp/mllm/vocab
adb push ../vocab/ling_vocab.mllm /data/local/tmp/mllm/vocab/
adb push ../vocab/ling_merges.txt /data/local/tmp/mllm/vocab/
adb push ../bin-arm/demo_bailing_moe /data/local/tmp/mllm/bin/
adb push ../bin-arm/demo_bailing_moe_mbp /data/local/tmp/mllm/bin/
adb push ../models/ling-lite-1.5-kai_q4_0.mllm /data/local/tmp/mllm/models/
# adb push ../models/ling-lite-1.5-kai_q4_0_e2.mllm /data/local/tmp/mllm/models/
# if push failed, exit
if [ $? -ne 0 ]; then
    echo "adb push failed"
    exit 1
fi
adb shell "cd /data/local/tmp/mllm/bin && ./demo_bailing_moe -m ../models/ling-lite-1.5-kai_q4_0.mllm"
adb shell "cd /data/local/tmp/mllm/bin && ./demo_bailing_moe_mbp -m ../models/ling-lite-1.5-kai_q4_0.mllm"
# adb shell "cd /data/local/tmp/mllm/bin && ./demo_bailing_moe -d 1 -m ../models/ling-lite-1.5-q4_0.mllm"
# adb shell "cd /data/local/tmp/mllm/bin && ./demo_bailing_moe -m ../models/ling-lite-1.5-kai_q4_0_e2.mllm"
# adb shell "cd /data/local/tmp/mllm/bin && ./demo_bailing_moe_mbp -m ../models/ling-lite-1.5-kai_q4_0_e2.mllm"