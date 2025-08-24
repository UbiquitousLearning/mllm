// Copyright (c) MLLM Team.
// Licensed under the MIT License.

package mllm

/*
#cgo CFLAGS: -fPIC -I${SRCDIR}/SDK/include/
#cgo LDFLAGS: -L${SRCDIR}/SDK/lib/
#cgo LDFLAGS: -lMllmSdkC
#cgo LDFLAGS: -Wl,-rpath=${SRCDIR}/SDK/lib

#include <mllm/mllm-c.hpp>
#include <stdlib.h>
*/
import "C"
import "fmt"

type ReturnCode int

const (
	Success ReturnCode = iota
	ErrorInvalidArgument
	ErrorOutOfMemory
	ErrorUnknown
)

type GenerationStatusCode int

const (
	GenerationSuccess GenerationStatusCode = iota
	GenerationEOF
)

type GenerationContext struct {
	ptr *C.struct_ARGenerationContext
}

func InitContext() ReturnCode {
	ret := C.mllm_init_context()
	return ReturnCode(ret)
}

func ShutdownContext() ReturnCode {
	ret := C.mllm_shutdown_context()
	return ReturnCode(ret)
}

func ShowMemoryReport() ReturnCode {
	ret := C.mllm_show_memory_report()
	return ReturnCode(ret)
}

func (r ReturnCode) Error() string {
	switch r {
	case ErrorInvalidArgument:
		return "invalid argument"
	case ErrorOutOfMemory:
		return "out of memory"
	case ErrorUnknown:
		return "unknown error"
	default:
		return fmt.Sprintf("unexpected error code: %d", r)
	}
}
