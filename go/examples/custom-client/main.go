// This example configures a client using credentials from custom environment variables.

package main

import (
	"context"
	"fmt"
	"log"
	"os"

	modal "github.com/modal-labs/modal-client/go"
)

func main() {
	ctx := context.Background()

	params := &modal.ClientParams{}
	if refreshToken := os.Getenv("CUSTOM_MODAL_OAUTH_REFRESH_TOKEN"); refreshToken != "" {
		clientID := os.Getenv("CUSTOM_MODAL_OAUTH_CLIENT_ID")
		if clientID == "" {
			log.Fatal("CUSTOM_MODAL_OAUTH_CLIENT_ID environment variable not set")
		}
		clientSecret := os.Getenv("CUSTOM_MODAL_OAUTH_CLIENT_SECRET")
		jwtKey := os.Getenv("CUSTOM_MODAL_OAUTH_JWT_KEY")
		if clientSecret == "" && jwtKey == "" {
			log.Fatal("set CUSTOM_MODAL_OAUTH_CLIENT_SECRET or CUSTOM_MODAL_OAUTH_JWT_KEY")
		}
		params.OAuthCredentials = &modal.OAuthCredentialsParams{
			RefreshToken: refreshToken,
			ClientID:     clientID,
			ClientSecret: clientSecret,
			JWTKey:       jwtKey,
		}
	} else {
		params.TokenID = os.Getenv("CUSTOM_MODAL_ID")
		if params.TokenID == "" {
			log.Fatal("CUSTOM_MODAL_ID environment variable not set")
		}
		params.TokenSecret = os.Getenv("CUSTOM_MODAL_SECRET")
		if params.TokenSecret == "" {
			log.Fatal("CUSTOM_MODAL_SECRET environment variable not set")
		}
	}

	mc, err := modal.NewClientWithOptions(params)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	echo, err := mc.Functions.FromName(ctx, "libmodal-test-support", "echo_string", nil)
	if err != nil {
		log.Fatalf("Failed to get Function: %v", err)
	}
	fmt.Printf("%#v\n", echo)
}
