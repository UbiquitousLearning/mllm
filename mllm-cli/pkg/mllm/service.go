// Copyright (c) MLLM Team.
// Licensed under the MIT License.
package mllm

import (
	"fmt"
	"log"
	"mllm-cli/mllm"
	"sync"
)

type Service struct {
	sessions map[string]*mllm.Session
	mutex    sync.Mutex
}

func NewService() *Service {
	return &Service{
		sessions: make(map[string]*mllm.Session),
	}
}

func (s *Service) RegisterSession(id string, session *mllm.Session) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	s.sessions[id] = session
}

func (s *Service) GetSession(sessionID string) (*mllm.Session, error) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if session, ok := s.sessions[sessionID]; ok {
		log.Printf("Found pre-registered session for model: %s", sessionID)
		return session, nil
	}

	return nil, fmt.Errorf("session for model '%s' not found. Is the server started with the correct --model-path?", sessionID)
}

func (s *Service) Shutdown() {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	log.Println("Shutting down all active sessions...")
	for id, session := range s.sessions {
		log.Printf("Closing session: %s", id)
		session.Close()
	}
	s.sessions = make(map[string]*mllm.Session)
	log.Println("All sessions have been shut down.")
}