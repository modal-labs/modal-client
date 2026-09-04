package test

import (
	"fmt"
	"os"
	"testing"
	"time"

	"go.uber.org/goleak"
)

// A Sandbox gives its connection back on an idle timer of its own rather than
// when the client is closed, so the tests run with a short one: the default
// leaves connections from the last few tests still open when the leak check
// runs, which is a leak that never happens and one that does look alike.
const testChannelIdleTimeout = "1"

func TestMain(m *testing.M) {
	err := os.Setenv("MODAL_ENVIRONMENT", "libmodal")
	if err != nil {
		panic(err)
	}
	err = os.Setenv("MODAL_SANDBOX_CHANNEL_IDLE_TIMEOUT", testChannelIdleTimeout)
	if err != nil {
		panic(err)
	}

	code := m.Run()

	// Give the last Sandbox's idle timer room to fire before looking for leaks.
	time.Sleep(3 * time.Second)
	if err := goleak.Find(); err != nil {
		fmt.Fprintf(os.Stderr, "goleak: %v\n", err)
		if code == 0 {
			code = 1
		}
	}
	os.Exit(code)
}
