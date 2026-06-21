// Copyright (c) MLLM Team.
// Licensed under the MIT License.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mllm-cli/pkg/api" 
	"net/http"
	"os"
	"strings"
)

func main() {
	serverURL := "http://localhost:8080/v1/chat/completions"
	var history []api.RequestMessage
	var currentSessionID string 

	fmt.Println("\n--- MLLM Refactored Interactive Client ---")
	fmt.Println("Supports multi-turn sessions. Type /exit to quit.")
	log.Printf("Connecting to: %s", serverURL)

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("\n> ")
		userInput, _ := reader.ReadString('\n')
		cleanedInput := strings.TrimSpace(userInput)
		if cleanedInput == "" { continue }
		if cleanedInput == "/exit" || cleanedInput == "/quit" { return }

		history = append(history, api.RequestMessage{Role: "user", Content: cleanedInput})
		apiRequest := api.OpenAIRequest{
			Model:     "Qwen3-0.6B-w4a32kai",
			Messages:  history,
			Stream:    true,
			SessionID: currentSessionID, 
		}
		requestBody, _ := json.Marshal(apiRequest)

		req, _ := http.NewRequest("POST", serverURL, bytes.NewBuffer(requestBody))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "text/event-stream")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			log.Printf("ERROR: Request failed: %v", err)
			history = history[:len(history)-1]
			continue
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			bodyBytes, _ := io.ReadAll(resp.Body)
			log.Printf("ERROR: Server returned status %s: %s", resp.Status, string(bodyBytes))
			history = history[:len(history)-1]
			continue
		}

		sessionIDFromHeader := resp.Header.Get("X-Session-ID")
		if sessionIDFromHeader != "" && currentSessionID != sessionIDFromHeader {
			currentSessionID = sessionIDFromHeader
			log.Printf("[Session Manager] New session established. ID: %s", currentSessionID)
		}

		var fullResponse strings.Builder
		scanner := bufio.NewScanner(resp.Body)
		fmt.Print("Assistant: ")
		for scanner.Scan() {
			line := scanner.Text()
			if strings.HasPrefix(line, "data: ") {
				jsonData := strings.TrimPrefix(line, "data: ")
				if jsonData == "[DONE]" { break }
				var chunk api.OpenAIResponseChunk
				if json.Unmarshal([]byte(jsonData), &chunk) == nil && len(chunk.Choices) > 0 {
					content := chunk.Choices[0].Delta.Content
					fmt.Print(content)
					fullResponse.WriteString(content)
				}
			}
		}
		fmt.Println()
		if err := scanner.Err(); err != nil { log.Printf("ERROR reading stream: %v", err) }
		history = append(history, api.RequestMessage{Role: "assistant", Content: fullResponse.String()})
	}
}