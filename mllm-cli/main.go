// Copyright (c) MLLM Team.
// Licensed under the MIT License.
package main

import (
	"mllm-cli/mllm"
)

func main() {
	mllm.InitializeContext()

	// // Run welcome interface first
	// if err := tui.RunWelcome(); err != nil {
	// 	fmt.Printf("Error running Welcome: %v\n", err)
	// 	return
	// }

	// // Then run model hub
	// selectedURL, err := tui.RunModelHub()
	// if err != nil {
	// 	fmt.Printf("Error running ModelHub: %v\n", err)
	// 	return
	// }
	// if selectedURL != "" {
	// 	fmt.Printf("Selected model URL: %s\n", selectedURL)
	// } else {
	// 	fmt.Println("No model selected.")
	// }
	mllm.ShutdownContext()
}
