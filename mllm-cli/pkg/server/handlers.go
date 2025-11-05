package server

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"

	"github.com/google/uuid"
)

func (s *Server) chatCompletionsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Only POST method is allowed", http.StatusMethodNotAllowed)
			return
		}

		var requestPayload map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&requestPayload); err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		modelName, _ := requestPayload["model"].(string)
		session, err := s.mllmService.GetSession(modelName)
		if err != nil {
			log.Printf("ERROR: Could not get session for model '%s': %v", modelName, err)
			http.Error(w, fmt.Sprintf("Model '%s' is not available on this server.", modelName), http.StatusNotFound)
			return
		}

		requestPayload["model"] = session.SessionID()

		requestID, ok := requestPayload["id"].(string)
		if !ok || requestID == "" {
			newID := uuid.New().String()
			log.Printf("Client did not provide a request ID. Generated a new one: %s", newID)
			requestID = newID
			requestPayload["id"] = newID
		}

		requestBytes, err := json.Marshal(requestPayload)
		if err != nil {
			http.Error(w, "Failed to re-marshal request payload", http.StatusInternalServerError)
			return
		}
		if !session.SendRequest(string(requestBytes)) {
			http.Error(w, "Failed to process request by the model", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		flusher, _ := w.(http.Flusher)

		log.Printf("Streaming response for session %s (Request ID: %s)...", session.SessionID(), requestID)
		
		for {
			if r.Context().Err() != nil {
				log.Printf("Client disconnected. Stopping stream for %s.", session.SessionID())
				break
			}

			
			rawResponse := session.PollResponse(requestID)

			if rawResponse == "" {
				log.Println("Received empty response from poll, assuming stream has ended.")
				break
			}
			
			fmt.Fprintf(w, "data: %s\n\n", rawResponse)
			flusher.Flush()

			var responseChunk map[string]interface{}
			if json.Unmarshal([]byte(rawResponse), &responseChunk) == nil {
				if choices, ok := responseChunk["choices"].([]interface{}); ok && len(choices) > 0 {
					if choice, ok := choices[0].(map[string]interface{}); ok {
						if reason, ok := choice["finish_reason"].(string); ok && reason == "stop" {
							log.Println("End of stream detected: finish_reason is 'stop'.")
							break
						}
					}
				}
			}
		}

		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
		log.Printf("Finished streaming for session %s (Request ID: %s).", session.SessionID(), requestID)
	}
}