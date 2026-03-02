package modal

import (
	"fmt"
	"time"
)

// RetriesParams are options for creating a Retries policy.
type RetriesParams struct {
	BackoffCoefficient *float32       // Multiplier for exponential backoff. Defaults to 2.0.
	InitialDelay       *time.Duration // Defaults to 1s.
	MaxDelay           *time.Duration // Defaults to 60s.
}

// Retries represents retry policy configuration for a Modal Function/Cls.
type Retries struct {
	MaxRetries         int
	BackoffCoefficient float32
	InitialDelay       time.Duration
	MaxDelay           time.Duration
}

// NewRetries creates a new Retries configuration.
func NewRetries(maxRetries int, params *RetriesParams) (*Retries, error) {
	backoffCoefficient := float32(2.0)
	initialDelay := 1 * time.Second
	maxDelay := 60 * time.Second

	if params != nil {
		if params.BackoffCoefficient != nil {
			backoffCoefficient = *params.BackoffCoefficient
		}
		if params.InitialDelay != nil {
			initialDelay = *params.InitialDelay
		}
		if params.MaxDelay != nil {
			maxDelay = *params.MaxDelay
		}
	}

	r := &Retries{
		MaxRetries:         maxRetries,
		BackoffCoefficient: backoffCoefficient,
		InitialDelay:       initialDelay,
		MaxDelay:           maxDelay,
	}

	if r.MaxRetries < 0 || r.MaxRetries > 10 {
		return nil, fmt.Errorf("invalid maxRetries: %d. Must be between 0 and 10", r.MaxRetries)
	}

	if r.BackoffCoefficient < 1.0 || r.BackoffCoefficient > 10.0 {
		return nil, fmt.Errorf("invalid backoffCoefficient: %f. Must be between 1.0 and 10.0", r.BackoffCoefficient)
	}

	if r.InitialDelay < 0 || r.InitialDelay > 60*time.Second {
		return nil, fmt.Errorf("invalid initialDelay: %v. Must be between 0 and 60s", r.InitialDelay)
	}

	if r.MaxDelay < 1*time.Second || r.MaxDelay > 60*time.Second {
		return nil, fmt.Errorf("invalid maxDelay: %v. Must be between 1s and 60s", r.MaxDelay)
	}

	return r, nil
}
