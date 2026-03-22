// Copyright (c) MLLM Team.
// Licensed under the MIT License.

package gomllm

import (
"log"
"os"
"path/filepath"

_ "golang.org/x/mobile/bind"
"mllm-cli/mllm"
pkgmllm "mllm-cli/pkg/mllm"
"mllm-cli/pkg/server"
)

func StartServer(modelPath string, ocrPath string, tmpDir string, enableProbing bool) string {
log.Println("[GoMobile] StartServer called")

if tmpDir != "" {
if err := os.Setenv("TMPDIR", tmpDir); err != nil {
log.Printf("[GoMobile] Error setting TMPDIR: %v", err)
} else {
log.Printf("[GoMobile] TMPDIR set to: %s", tmpDir)
}
}

if !mllm.InitializeContext() {
return "Error: InitializeContext failed"
}
mllm.SetLogLevel(2)

service := pkgmllm.NewService()

if modelPath != "" {
log.Printf("[GoMobile] Loading Qwen: %s", modelPath)
probePath := filepath.Join(modelPath, "probes_linear")
var (
session *mllm.Session
err     error
)
if enableProbing {
if stat, statErr := os.Stat(probePath); statErr == nil && stat.IsDir() {
log.Printf("[GoMobile] Probing enabled, probes found: %s", probePath)
session, err = mllm.NewProbingSession(modelPath, probePath)
} else {
log.Printf("[GoMobile] Probes not found, fallback to normal Qwen session. expected=%s", probePath)
session, err = mllm.NewSession(modelPath)
}
} else {
log.Printf("[GoMobile] Probing disabled. Using normal Qwen session.")
session, err = mllm.NewSession(modelPath)
}
if err != nil {
return "Error: Qwen load failed: " + err.Error()
}
sessionID := filepath.Base(modelPath)
session.Insert(sessionID)
service.RegisterSession(sessionID, session)
}

if ocrPath != "" {
log.Printf("[GoMobile] Loading OCR: %s", ocrPath)
session, err := mllm.NewDeepseekOCRSession(ocrPath)
if err != nil {
return "Error: OCR load failed: " + err.Error()
}
sessionID := filepath.Base(ocrPath)
session.Insert(sessionID)
service.RegisterSession(sessionID, session)
}

if !mllm.StartService(1) {
return "Error: StartService failed"
}

go func() {
s := server.NewServer("127.0.0.1:8080", service)
log.Println("[GoMobile] HTTP Server listening on 8080")
s.Start()
}()

return "Success: Server Running on 127.0.0.1:8080"
}
