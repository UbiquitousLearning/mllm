// Copyright (c) MLLM Team.
// Licensed under the MIT License.
package main

import (
	"mllm-cli/mllm"
)

func main() {
	mllm.InitContext()

	mllm.ShutdownContext()
}
