package modal

// config.go houses the logic for loading and resolving Modal profiles
// from ~/.modal.toml or environment variables.

import (
	"errors"
	"fmt"
	"math"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"time"

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
	// MaxThrottleWait controls server-driven (throttle) retries.
	// nil = no limit; 0 = disable server-driven retries entirely; >0 = cap total wait to this many seconds.
	MaxThrottleWait *time.Duration
	// SandboxChannelIdleTimeout is how long a Sandbox connection may sit idle
	// before the client gives it up. The Sandbox stays usable: the next
	// operation reconnects. Zero keeps connections open until the client closes.
	SandboxChannelIdleTimeout time.Duration
	// SandboxStreamIdleTimeout is how long a caller may sit on a chunk of a
	// Sandbox's output before the stream stops counting as in use. Only once a
	// reader has gone quiet for this long does the idle timeout above start
	// running, so a Sandbox read once and then forgotten gives its connection up
	// after the two together. Zero stops counting as soon as a caller stops
	// reading.
	SandboxStreamIdleTimeout time.Duration
}

const (
	// How long a Sandbox connection may sit idle before it is released, unless
	// MODAL_SANDBOX_CHANNEL_IDLE_TIMEOUT says otherwise.
	defaultSandboxChannelIdleTimeout = 30 * time.Second
	// How long a caller may sit on a chunk before their stream stops counting as
	// in use, unless MODAL_SANDBOX_STREAM_IDLE_TIMEOUT says otherwise.
	defaultSandboxStreamIdleTimeout = 5 * time.Second
)

// maxIdleTimeoutSeconds is as long as an idle timeout can be said in seconds
// and still convert to a time.Duration. Dividing as integers keeps the bound
// under the point where the conversion goes out of range, which Go leaves to
// the platform: some saturate, others hand back a negative.
const maxIdleTimeoutSeconds = float64(math.MaxInt64 / int64(time.Second))

// parseIdleTimeoutSeconds reads a non-negative number of seconds, reporting
// whether it was one. Infinity and NaN parse as floats but are not durations,
// and neither is a number too large to hold, so all three are refused rather
// than turned into a nonsense deadline.
func parseIdleTimeoutSeconds(s string) (time.Duration, bool) {
	v, err := strconv.ParseFloat(s, 64)
	if err != nil || !(v >= 0) || v > maxIdleTimeoutSeconds {
		return 0, false
	}
	d := time.Duration(v * float64(time.Second))
	if v > 0 && d == 0 {
		// Zero is how a caller turns the release off, so a timeout too short to
		// measure becomes the shortest one there is rather than none at all.
		return 1, true
	}
	return d, true
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

	var maxThrottleWait *time.Duration
	if s := os.Getenv("MODAL_MAX_THROTTLE_WAIT"); s != "" {
		if v, err := strconv.ParseInt(s, 10, 64); err == nil && v >= 0 {
			d := time.Duration(v) * time.Second
			maxThrottleWait = &d
		}
	}

	sandboxChannelIdleTimeout := defaultSandboxChannelIdleTimeout
	if s := os.Getenv("MODAL_SANDBOX_CHANNEL_IDLE_TIMEOUT"); s != "" {
		if v, ok := parseIdleTimeoutSeconds(s); ok {
			sandboxChannelIdleTimeout = v
		}
	}

	sandboxStreamIdleTimeout := defaultSandboxStreamIdleTimeout
	if s := os.Getenv("MODAL_SANDBOX_STREAM_IDLE_TIMEOUT"); s != "" {
		if v, ok := parseIdleTimeoutSeconds(s); ok {
			sandboxStreamIdleTimeout = v
		}
	}

	return Profile{
		ServerURL:           serverURL,
		TokenID:             tokenID,
		TokenSecret:         tokenSecret,
		Environment:         environment,
		ImageBuilderVersion: imageBuilderVersion,
		LogLevel:            logLevel,
		MaxThrottleWait:     maxThrottleWait,

		SandboxChannelIdleTimeout: sandboxChannelIdleTimeout,
		SandboxStreamIdleTimeout:  sandboxStreamIdleTimeout,
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
