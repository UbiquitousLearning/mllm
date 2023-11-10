import subprocess
import sys
import json

Device_List = []
if __name__ == "__main__":
    output = subprocess.getoutput("adb --json devices -l")
    devices = json.loads(output)
    # [{"serial":"6fd28b00","state":"Device","product":"syberia_oneplus6","model":"ONEPLUS_A6000","device":"oneplus6","devpath":"usb:3-4","transport_id":1}]
    Device_List = list(
        filter(lambda device: device.get("state", None) == "Device", devices)
    )
    for device in Device_List:
        print(device)
        subprocess.run(
            f"adb -s {device.get('serial','')} shell -- rm -rf /data/local/tmp/mllm",
            shell=True,
        )
        subprocess.run(
            f"adb -s {device.get('serial','')} push ./bin/ /data/local/tmp/mllm/",
            shell=True,
        )
        subprocess.run(
            f"adb -s {device.get('serial','')} push ./bin-arm/ /data/local/tmp/mllm/",
            shell=True,
        )
        subprocess.run(
            f"adb -s {device.get('serial','')} shell -- chmod +x /data/local/tmp/mllm/MLLM_TEST",
            shell=True,
        )
        subprocess.run(
            f"adb -s {device.get('serial','')} shell  -- cd /data/local/tmp/mllm \"&&\" LD_LIBRARY_PATH=/data/local/tmp/:$LD_LIBRARY_PATH /data/local/tmp/mllm/MLLM_TEST \"--gtest_filter=*\" \"--gtest_output=json\"",
            #    Output the stdout
            shell=True,
        )
        subprocess.run(
            f"adb -s {device.get('serial','')} pull /data/local/tmp/mllm/test_detail.json ./adb_{device.get('device','')}.json",
            shell=True,
        )
