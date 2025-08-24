// Copyright (c) MLLM Team.
// Licensed under the MIT License.

package mllm

/*
#cgo CFLAGS: -fPIC -I${SRCDIR}/SDK/include/
#cgo LDFLAGS: -L${SRCDIR}/SDK/lib/
#cgo LDFLAGS: -lMllmSdkC
#cgo LDFLAGS: -Wl,-rpath ${SRCDIR}/SDK/lib

#include <mllm/mllm-c.hpp>
#include <stdlib.h>
*/
import "C"

// InitMllmContext initialize the MLLM context
func InitContext() int {
	return int(C.mllm_init_context())
}

// ShutdownMllmContext shutdown the MLLM context
func ShutdownContext() int {
	return int(C.mllm_shutdown_context())
}

func ShowMemoryReport() int {
	return int(C.mllm_show_memory_report())
}
