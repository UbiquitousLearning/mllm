# FastRPC session redirect

This preload library reserves a new CDSP session and redirects ordinary
session-0 FastRPC opens, mappings, controls, and DSP queues to it. It is used
by the reproducible benchmark when CDSP session 0 is unavailable; it does not
reset the DSP or reboot the phone.

Build an Android arm64 library with:

```bash
ANDROID_NDK=/path/to/android-ndk \
HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK \
tools/fastrpc_session_redirect/build_android.sh
```

Preload the result only for the benchmark process. A successful run prints
`FastRPC redirect: CDSP domain 3/session 0 -> ...` before opening QNN/HMX.
