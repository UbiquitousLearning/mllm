// Copyright (c) MLLM Team.
// Licensed under the MIT License.

package mllm

/*
#cgo CFLAGS: -fPIC -I${SRCDIR}/SDK/include/
#cgo CFLAGS: -std=c11
#cgo LDFLAGS: -L${SRCDIR}/SDK/lib/
#cgo LDFLAGS: -lMllmSdkC
#cgo LDFLAGS: -Wl,-rpath ${SRCDIR}/SDK/lib

#include <mllm/mllm-c.h>
#include <stdlib.h>
*/
import "C"

func isOk(any C.MllmCAny) bool {
	return C.isOk(any) == 0
}

func InitializeContext() bool {
	return isOk(C.initializeContext())
}

func ShutdownContext() bool {
	return isOk(C.shutdownContext())
}
