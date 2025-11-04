// Copyright (c) MLLM Team.
// Licensed under the MIT License.
package main

import (
	"context"
	"flag"
	"log"
	"mllm-cli/mllm"
	pkgmllm "mllm-cli/pkg/mllm"
	"mllm-cli/pkg/server"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"
)

func main() {
	modelPath := flag.String("model-path", "", "Path to the MLLM model directory.")
	flag.Parse()

	if *modelPath == "" {
		log.Fatal("FATAL: --model-path argument is required.")
	}

	if !mllm.InitializeContext() {
		log.Fatal("FATAL: InitializeContext failed!")
	}
	mllm.SetLogLevel(2)
	if !mllm.StartService(4) {
		log.Fatal("FATAL: StartService failed!")
	}
	defer mllm.StopService()
	defer mllm.ShutdownContext()

	mllmService := pkgmllm.NewService()

	log.Printf("Loading model and creating session from: %s", *modelPath)
	session, err := mllm.NewSession(*modelPath)
	if err != nil {
		log.Fatalf("FATAL: Failed to create session: %v", err)
	}

	sessionID := filepath.Base(*modelPath)
	if !session.Insert(sessionID) {
		session.Close()
		log.Fatalf("FATAL: Failed to insert session with ID '%s'", sessionID)
	}
	mllmService.RegisterSession(sessionID, session)
	log.Printf("Session created and registered successfully with ID: %s", sessionID)

	httpServer := server.NewServer(":8080", mllmService)

	go httpServer.Start()

	shutdownChan := make(chan os.Signal, 1)
	signal.Notify(shutdownChan, syscall.SIGINT, syscall.SIGTERM)
	<-shutdownChan

	log.Println("Received shutdown signal. Starting graceful shutdown...")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := httpServer.Shutdown(ctx); err != nil {
		log.Printf("HTTP server shutdown failed: %v", err)
	}

	mllmService.Shutdown()

	log.Println("Server gracefully stopped.")
}