// Copyright (c) MLLM Team.
// Licensed under the MIT License.
package api

type RequestMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OpenAIRequest struct {
	Model          string           `json:"model"`
	Messages       []RequestMessage `json:"messages"`
	Stream         bool             `json:"stream"`
	EnableThinking bool             `json:"enable_thinking,omitempty"` 
	Thinking       bool             `json:"thinking,omitempty"`       // <-- 新增此行，用于接收客户端可能发送的 "thinking": true
	SessionID      string           `json:"session_id,omitempty"`     
}

type ResponseDelta struct {
	Content string `json:"content"`
}

type ResponseChoice struct {
	Delta ResponseDelta `json:"delta"`
}

type OpenAIResponseChunk struct {
	ID      string           `json:"id"`
	Object  string           `json:"object"`
	Created int64            `json:"created"`
	Model   string           `json:"model"`
	Choices []ResponseChoice `json:"choices"`
}