package main

import (
	"fmt"

	modal "github.com/modal-labs/modal-client/go"
)

func main() {
	client, err := modal.NewClient()
	if err != nil {
		panic(fmt.Sprintf("ERROR: Failed to create client: %v", err))
	}
	defer client.Close()

	fmt.Println(client.Version())
}
