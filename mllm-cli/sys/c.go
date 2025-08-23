// Copyright (c) MLLM Team.
// Licensed under the MIT License.

package sys

/*
#cgo CXXFLAGS: -std=c++20 -fPIC -I../../mllm
#cgo darwin LDFLAGS: -lMllmRT
#cgo linux LDFLAGS: -lMllmRT
#cgo windows LDFLAGS: -lMllmRT

#include "mllm-c.h"
#include <stdlib.h>
*/
import "C"

// InitContext initialize the MLLM context
func InitContext() int {
	return int(C.mllm_init_context())
}

// ShutdownContext shutdown the MLLM context
func ShutdownContext() int {
	return int(C.mllm_shutdown_context())
}
