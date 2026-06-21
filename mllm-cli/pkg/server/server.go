// Copyright (c) MLLM Team.
// Licensed under the MIT License.
package server

import (
	"context"
	"log"
	"mllm-cli/pkg/mllm"
	"net/http"
)

type Server struct {
	httpServer  *http.Server
	mllmService *mllm.Service
}

func NewServer(addr string, mllmService *mllm.Service) *Server {
	mux := http.NewServeMux()
	
	s := &Server{
		httpServer: &http.Server{
			Addr:    addr,
			Handler: mux,
		},
		mllmService: mllmService,
	}

	mux.HandleFunc("/v1/chat/completions", s.chatCompletionsHandler())
	return s
}

func (s *Server) Start() {
	log.Printf("OpenAI-compatible API server listening on %s", s.httpServer.Addr)
	if err := s.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("FATAL: Could not start HTTP server: %v", err)
	}
}

func (s *Server) Shutdown(ctx context.Context) error {
	log.Println("Shutting down HTTP server...")
	return s.httpServer.Shutdown(ctx)
}