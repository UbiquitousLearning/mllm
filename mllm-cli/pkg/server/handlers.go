// Copyright (c) MLLM Team.
// Licensed under the MIT License.
package server

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/google/uuid"
)

func decodeBase64Image(uri string) ([]byte, string, error) {
	if !strings.HasPrefix(uri, "data:image/") {
		return nil, "", fmt.Errorf("invalid data URI: must start with 'data:image/'")
	}

	parts := strings.SplitN(uri, ",", 2)
	if len(parts) != 2 {
		return nil, "", fmt.Errorf("invalid base64 image data")
	}
	
	meta := parts[0]
	ext := "" 
	if strings.Contains(meta, "image/jpeg") {
		ext = ".jpg"
	} else if strings.Contains(meta, "image/webp") {
		ext = ".webp"
	} else if strings.Contains(meta, "image/png") {
		ext = ".png"
	} else {
		return nil, "", fmt.Errorf("unsupported image format in data URI")
	}

	data, err := base64.StdEncoding.DecodeString(parts[1])
	if err != nil {
		return nil, "", fmt.Errorf("failed to decode base64: %v", err)
	}
	return data, ext, nil
}

func (s *Server) preprocessRequestForOCR(payload map[string]interface{}) (bool, func(), error) {
	messages, ok := payload["messages"].([]interface{})
	if !ok {
		return false, nil, fmt.Errorf("invalid messages format")
	}

	var userMessage map[string]interface{}
	var contentArray []interface{}
	var imageFoundInPayload bool = false

	for i := len(messages) - 1; i >= 0; i-- {
		msg, ok := messages[i].(map[string]interface{})
		if !ok {
			continue
		}
		if role, _ := msg["role"].(string); role == "user" {
			delete(msg, "images") 

			if content, ok := msg["content"].([]interface{}); ok {
				userMessage = msg
				contentArray = content
				log.Println("[Handler] Found OpenAI Vision 'content' array.")
				break
			} else if images, ok := msg["images"].([]interface{}); ok && len(images) > 0 {
				log.Println("[Handler] Found custom 'images' field.")
				base64URI, ok := images[0].(string)
				if !ok || !strings.HasPrefix(base64URI, "data:image") {
					return false, func() {}, fmt.Errorf("image data in 'images' field is not a valid base64 URI")
				}

				textContent, _ := msg["content"].(string)

				contentArray = []interface{}{
					map[string]interface{}{"type": "text", "text": textContent},
					map[string]interface{}{"type": "image_url", "image_url": map[string]interface{}{"url": base64URI}},
				}
				userMessage = msg
				break
			} else {
				userMessage = msg
				contentArray = nil
				log.Println("[Handler] Found text-only user message.")
				break
			}
		}
	}

	if userMessage == nil {
		return false, nil, fmt.Errorf("no user message found")
	}

	var textContent string
	var base64URI string
	if contentArray != nil {
		for _, part := range contentArray {
			partMap, ok := part.(map[string]interface{})
			if !ok {
				continue
			}
			partType, _ := partMap["type"].(string)
			if partType == "text" {
				textContent, _ = partMap["text"].(string)
			} else if partType == "image_url" {
				imageUrl, ok := partMap["image_url"].(map[string]interface{})
				if ok {
					base64URI, _ = imageUrl["url"].(string)
					imageFoundInPayload = true 
				}
			}
		}
	} else {
		textContent, _ = userMessage["content"].(string)
	}

    if strings.TrimSpace(textContent) == "" {
        log.Println("[Handler] User content is empty, auto-filling default prompt.")
        textContent = "Convert the document to markdown."
    }

	if !imageFoundInPayload {
		log.Println("[Handler] No new image found in payload for OCR request.")
		userMessage["content"] = textContent 
		delete(userMessage, "images") 
		return false, func() {}, nil 
	}

	log.Println("[Handler] New image found. Processing...")

	imageData, ext, err := decodeBase64Image(base64URI)
	if err != nil {
		return false, nil, err
	}
	tempFile, err := os.CreateTemp("", "ocr_temp_*"+ext)
	if err != nil {
		return false, nil, fmt.Errorf("failed to create temp file: %v", err)
	}
	if _, err := tempFile.Write(imageData); err != nil {
		tempFile.Close()
		os.Remove(tempFile.Name())
		return false, nil, fmt.Errorf("failed to write to temp file: %v", err)
	}
	tempFile.Close()
	absPath, err := filepath.Abs(tempFile.Name())
	if err != nil {
		os.Remove(tempFile.Name())
		return false, nil, fmt.Errorf("failed to get absolute path for temp file: %v", err)
	}
	log.Printf("[Handler] Saved Base64 image to temporary file: %s", absPath)

	userMessage["content"] = textContent
	userMessage["images"] = []interface{}{absPath}

	cleanupFunc := func() {
		log.Printf("[Handler] Cleaning up temporary file: %s", absPath)
		os.Remove(absPath)
	}
	return true, cleanupFunc, nil
}

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

		if strings.Contains(strings.ToLower(modelName), "ocr") || strings.HasSuffix(strings.ToLower(modelName), "-ocr") {
			log.Printf("[Handler] OCR model detected ('%s'). Checking for image data...", modelName)
			
			imageFound, cleanupFunc, err := s.preprocessRequestForOCR(requestPayload)
			if err != nil {
				log.Printf("ERROR: Failed to process OCR request: %v", err)
				http.Error(w, fmt.Sprintf("Failed to process OCR request: %v", err), http.StatusBadRequest)
				return
			}
			defer cleanupFunc()

			if !imageFound {
				log.Println("ERROR: OCR model is single-turn and requires an image in *every* request. Text-only follow-ups are not supported.")
				http.Error(w, "OCR model is single-turn and requires an image in every request. Text-only follow-ups are not supported.", http.StatusBadRequest)
				return
			}

		} else {
			log.Printf("[Handler] Text model detected ('%s'). Forwarding request...", modelName)
		}

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
			if fl, ok := w.(http.Flusher); ok {
				fl.Flush()
			}

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
		if fl, ok := w.(http.Flusher); ok {
			fl.Flush()
		}
		log.Printf("Finished streaming for session %s (Request ID: %s).", session.SessionID(), requestID)
	}
}