#!/bin/bash
# NOTE: just for dev convinience
adb push ../bin/nnapi_test /data/local/tmp
# if push failed, exit
if [ $? -ne 0 ]; then
    echo "adb push failed"
    exit 1
fi
adb shell ./data/local/tmp/nnapi_test