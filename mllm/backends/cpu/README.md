# CPU Backend


## Dispatcher Behavior

If the `Module` is executed directly, the threads opened by the `Ops` in the `Module` will utilize the current main thread (due to OpenMP). If the `Module` is executed in async mode, the threads opened by the `Ops` in the `Module` are independent of the main thread.
