package modal

import (
	"errors"
	"os"
	"path/filepath"

	"github.com/BurntSushi/toml"
)

type Config struct {
	Active      *bool   `toml:"active"`
	ServerURL   *string `toml:"server_url"`
	TokenID     *string `toml:"token_id"`
	TokenSecret *string `toml:"token_secret"`
	TaskID      *string `toml:"task_id"`
	TaskSecret  *string `toml:"task_secret"`
}

type Configs map[string]Config

func LoadConfig() (*Config, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return nil, errors.New("HOME env var not set!")
	}

	contents, err := os.ReadFile(filepath.Join(homeDir, ".modal.toml"))
	if err != nil {
		return nil, err
	}

	var configs Configs
	if err := toml.Unmarshal(contents, &configs); err != nil {
		return nil, err
	}

	for _, config := range configs {
		if config.Active != nil && *config.Active {
			return &config, nil
		}
	}

	return nil, errors.New("no active config found")
}
