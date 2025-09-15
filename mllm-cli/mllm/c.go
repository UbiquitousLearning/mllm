// Copyright (c) MLLM Team.
// Licensed under the MIT License.

package mllm

/*
#cgo CFLAGS: -fPIC -I${SRCDIR}/SDK/include/
#cgo CFLAGS: -std=c11
#cgo LDFLAGS: -L${SRCDIR}/SDK/lib/
#cgo LDFLAGS: -lMllmSdkC
#cgo LDFLAGS: -Wl,-rpath ${SRCDIR}/SDK/lib

#include <mllm/mllm-c.hpp>
#include <stdlib.h>
*/
import "C"

func InitContext() {
	ret := C.mllm_ret_is_success(C.mllm_init_context())
	if ret != 0 {
		panic("Failed to initialize MLLM context")
	}
}

func ShutdownContext() {
	C.mllm_shutdown_context()
}
