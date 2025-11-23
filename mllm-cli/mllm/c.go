// Copyright (c) MLLM Team.
// Licensed under the MIT License.

package mllm

/*
#cgo CFLAGS: -std=c11
#cgo LDFLAGS: -lMllmSdkC -lMllmRT -lMllmCPUBackend

#include <mllm/mllm-c.h>
#include <stdlib.h>

static void* MllmCAny_get_v_custom_ptr(MllmCAny handle) {
    return handle.v_custom_ptr;
}

static MllmCAny MllmCAny_set_v_custom_ptr_null(MllmCAny handle) {
    handle.v_custom_ptr = NULL;
    return handle;
}
*/
import "C"
import "unsafe"
import "fmt"
import "runtime"


type Session struct {
    cHandle C.MllmCAny
    sessionID string
}

func isOk(any C.MllmCAny) bool {
	return C.isOk(any) == 1
}

func InitializeContext() bool {
	return isOk(C.initializeContext())
}

func ShutdownContext() bool {
	return isOk(C.shutdownContext())
}

func StartService(workerThreads int) bool {
    result := C.startService(C.size_t(workerThreads))
    return isOk(result)
}

func StopService() bool {
    result := C.stopService()
    return isOk(result)
}

func SetLogLevel(level int) {
    C.setLogLevel(C.int(level))
}

func NewSession(modelPath string) (*Session, error) {
    cModelPath := C.CString(modelPath)
    defer C.free(unsafe.Pointer(cModelPath))

    handle := C.createQwen3Session(cModelPath)
    if !isOk(handle) {
        return nil, fmt.Errorf("底层C API createQwen3Session 失败")
    }
    s := &Session{cHandle: handle}
    runtime.SetFinalizer(s, func(s *Session) {
        fmt.Println("[Go Finalizer] Mllm Session automatically released.") 
        C.freeSession(s.cHandle)
    })

    return s, nil
}

func NewDeepseekOCRSession(modelPath string) (*Session, error) {
    cModelPath := C.CString(modelPath)
    defer C.free(unsafe.Pointer(cModelPath))

    handle := C.createDeepseekOCRSession(cModelPath) 
    if !isOk(handle) { 
        return nil, fmt.Errorf("底层C API createDeepseekOCRSession 失败")
    }
    s := &Session{cHandle: handle} 
    runtime.SetFinalizer(s, func(s *Session) {
        fmt.Println("[Go Finalizer] Mllm OCR Session automatically released.")
        C.freeSession(s.cHandle)
    })

    return s, nil
}

func (s *Session) Close() {
    if C.MllmCAny_get_v_custom_ptr(s.cHandle) != nil {
        fmt.Println("[Go Close] Mllm Session manually closed.") 
        C.freeSession(s.cHandle)
        s.cHandle = C.MllmCAny_set_v_custom_ptr_null(s.cHandle)
        runtime.SetFinalizer(s, nil)
    }
}

func (s *Session) Insert(sessionID string) bool {
    cSessionID := C.CString(sessionID)
    defer C.free(unsafe.Pointer(cSessionID))
    result := C.insertSession(cSessionID, s.cHandle)
    if isOk(result) {
        s.sessionID = sessionID 
    }
    return isOk(result)
}

func (s *Session) SendRequest(jsonRequest string) bool {
    if s.sessionID == "" {
        fmt.Println("[Go SendRequest] Error: sessionID is not set on this session.")
        return false 
    }
    cSessionID := C.CString(s.sessionID)
    cJsonRequest := C.CString(jsonRequest)
    defer C.free(unsafe.Pointer(cSessionID))
    defer C.free(unsafe.Pointer(cJsonRequest))

    result := C.sendRequest(cSessionID, cJsonRequest)
    return isOk(result)
}

func (s *Session) PollResponse(requestID string) string { 
    if requestID == "" {
        fmt.Println("[Go PollResponse] Error: requestID cannot be empty.")
        return ""
    }
    cRequestID := C.CString(requestID)
    defer C.free(unsafe.Pointer(cRequestID))

    cResponse := C.pollResponse(cRequestID) 
    if cResponse == nil {
        return ""
    }
    defer C.freeResponseString(cResponse)
    
    return C.GoString(cResponse)
}

func (s *Session) SessionID() string {
    return s.sessionID
}