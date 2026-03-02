package modal

// config.go houses the logic for loading and resolving Modal profiles
// from ~/.modal.toml or environment variables.

import (
	"errors"
	"fmt"
	"net/url"
	"os"
	"path/filepath"

	"github.com/pelletier/go-toml/v2"
)

// Profile holds a fully-resolved configuration ready for use by the client.
type Profile struct {
	ServerURL           string
	TokenID             string
	TokenSecret         string
	Environment         string
	ImageBuilderVersion string
	LogLevel            string
}

func (p Profile) isLocalhost() bool {
	parsedURL, err := url.Parse(p.ServerURL)
	if err != nil {
		return false
	}
	hostname := parsedURL.Hostname()
	return hostname == "localhost" || hostname == "127.0.0.1" || hostname == "::1" || hostname == "172.21.0.1"
}

// rawProfile mirrors the TOML structure on disk.
type rawProfile struct {
	ServerURL           string `toml:"server_url"`
	TokenID             string `toml:"token_id"`
	TokenSecret         string `toml:"token_secret"`
	Environment         string `toml:"environment"`
	ImageBuilderVersion string `toml:"image_builder_version"`
	LogLevel            string `toml:"loglevel"`
	Active              bool   `toml:"active"`
}

type config map[string]rawProfile

func configFilePath() (string, error) {
	if configPath := os.Getenv("MODAL_CONFIG_PATH"); configPath != "" {
		return configPath, nil
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("cannot locate homedir: %w", err)
	}
	return filepath.Join(home, ".modal.toml"), nil
}

// readConfigFile loads the Modal config file, returning an empty config if the file
// does not exist.
func readConfigFile() (config, error) {
	path, err := configFilePath()
	if err != nil {
		return nil, err
	}

	content, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return config{}, nil // silent absence is fine
	} else if err != nil {
		return nil, fmt.Errorf("reading %s: %w", path, err)
	}

	var cfg config
	if err := toml.Unmarshal(content, &cfg); err != nil {
		return nil, fmt.Errorf("parsing %s: %w", path, err)
	}
	return cfg, nil
}

// getProfile resolves a profile by name. Pass an empty string to instead return
// the first profile in the configuration file with `active = true`.
//
// Returned Profile is ready for use; error describes what is missing.
func getProfile(name string, cfg config) Profile {
	if name == "" {
		for n, p := range cfg {
			if p.Active {
				name = n
				break
			}
		}
	}

	var raw rawProfile
	if name != "" {
		raw = cfg[name]
	}

	// Env-vars override file values.
	serverURL := firstNonEmpty(os.Getenv("MODAL_SERVER_URL"), raw.ServerURL, "https://api.modal.com:443")
	tokenID := firstNonEmpty(os.Getenv("MODAL_TOKEN_ID"), raw.TokenID)
	tokenSecret := firstNonEmpty(os.Getenv("MODAL_TOKEN_SECRET"), raw.TokenSecret)
	environment := firstNonEmpty(os.Getenv("MODAL_ENVIRONMENT"), raw.Environment)
	imageBuilderVersion := firstNonEmpty(os.Getenv("MODAL_IMAGE_BUILDER_VERSION"), raw.ImageBuilderVersion)
	logLevel := firstNonEmpty(os.Getenv("MODAL_LOGLEVEL"), raw.LogLevel)

	return Profile{
		ServerURL:           serverURL,
		TokenID:             tokenID,
		TokenSecret:         tokenSecret,
		Environment:         environment,
		ImageBuilderVersion: imageBuilderVersion,
		LogLevel:            logLevel,
	}
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

func environmentName(environment string, profile Profile) string {
	return firstNonEmpty(environment, profile.Environment)
}

func imageBuilderVersion(version string, profile Profile) string {
	return firstNonEmpty(version, profile.ImageBuilderVersion, "2024.10")
}
