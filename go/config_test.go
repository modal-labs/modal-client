package modal

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/onsi/gomega"
)

func TestGetConfigPath_WithEnvVar(t *testing.T) {
	g := gomega.NewWithT(t)

	customPath := "/custom/path/to/config.toml"
	t.Setenv("MODAL_CONFIG_PATH", customPath)

	path, err := configFilePath()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(path).Should(gomega.Equal(customPath))
}

func TestGetConfigPath_WithoutEnvVar(t *testing.T) {
	g := gomega.NewWithT(t)

	t.Setenv("MODAL_CONFIG_PATH", "")

	path, err := configFilePath()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	home, _ := os.UserHomeDir()
	expectedPath := filepath.Join(home, ".modal.toml")
	g.Expect(path).Should(gomega.Equal(expectedPath))
}

func TestGetProfile_MaxThrottleWaitParsing(t *testing.T) {
	cases := []struct {
		envVal   string
		expected *time.Duration
	}{
		{"10", durationPtr(10 * time.Second)},
		{"0", durationPtr(0)},
		{"3600", durationPtr(3600 * time.Second)},
	}

	for _, tc := range cases {
		t.Run(tc.envVal, func(t *testing.T) {
			g := gomega.NewWithT(t)
			t.Setenv("MODAL_MAX_THROTTLE_WAIT", tc.envVal)
			profile := getProfile("", config{})
			g.Expect(profile.MaxThrottleWait).NotTo(gomega.BeNil())
			g.Expect(*profile.MaxThrottleWait).To(gomega.Equal(*tc.expected))
		})
	}
}

func durationPtr(d time.Duration) *time.Duration { return &d }

func TestGetProfile_MaxThrottleWaitInvalidValue(t *testing.T) {
	g := gomega.NewWithT(t)

	t.Setenv("MODAL_MAX_THROTTLE_WAIT", "not-a-number")

	profile := getProfile("", config{})
	g.Expect(profile.MaxThrottleWait).To(gomega.BeNil())
}

func TestProfileIsLocalhost(t *testing.T) {
	g := gomega.NewWithT(t)
	p := Profile{ServerURL: "http://localhost:8889"}
	g.Expect(p.isLocalhost()).Should(gomega.BeTrue())
}

// A value that parses as a float is not necessarily a duration. Infinity, NaN
// and anything too large to hold must fall back rather than become a nonsense
// deadline.
func TestGetProfile_IdleTimeoutRejectsValuesThatAreNotDurations(t *testing.T) {
	defaults := getProfile("", config{})

	cases := []struct {
		envVar string
		want   time.Duration
	}{
		{"MODAL_SANDBOX_CHANNEL_IDLE_TIMEOUT", defaults.SandboxChannelIdleTimeout},
		{"MODAL_SANDBOX_STREAM_IDLE_TIMEOUT", defaults.SandboxStreamIdleTimeout},
	}
	bad := []string{"Inf", "+Inf", "Infinity", "-Inf", "NaN", "1e30", "-1", "nonsense"}

	for _, tc := range cases {
		for _, value := range bad {
			t.Run(tc.envVar+"="+value, func(t *testing.T) {
				g := gomega.NewWithT(t)
				t.Setenv(tc.envVar, value)

				profile := getProfile("", config{})
				got := profile.SandboxChannelIdleTimeout
				if tc.envVar == "MODAL_SANDBOX_STREAM_IDLE_TIMEOUT" {
					got = profile.SandboxStreamIdleTimeout
				}
				g.Expect(got).To(gomega.Equal(tc.want), "should have fallen back to the default")
			})
		}
	}
}

// A positive timeout too short to measure must not read as zero, which is how
// the release is turned off - that would invert what was asked for.
func TestGetProfile_IdleTimeoutKeepsAShortTimeoutPositive(t *testing.T) {
	g := gomega.NewWithT(t)

	t.Setenv("MODAL_SANDBOX_CHANNEL_IDLE_TIMEOUT", "0.0000000001")
	t.Setenv("MODAL_SANDBOX_STREAM_IDLE_TIMEOUT", "0.0000000001")

	profile := getProfile("", config{})
	g.Expect(profile.SandboxChannelIdleTimeout).To(gomega.BeNumerically(">", 0))
	g.Expect(profile.SandboxStreamIdleTimeout).To(gomega.BeNumerically(">", 0))
}

// The largest accepted value still converts to a positive duration. Go leaves
// an out-of-range float-to-int conversion to the platform, so a bound that
// allowed one would be negative on some and saturated on others.
func TestGetProfile_IdleTimeoutAtTheUpperBoundStaysPositive(t *testing.T) {
	g := gomega.NewWithT(t)

	t.Setenv("MODAL_SANDBOX_CHANNEL_IDLE_TIMEOUT", strconv.FormatFloat(maxIdleTimeoutSeconds, 'f', -1, 64))

	profile := getProfile("", config{})
	g.Expect(profile.SandboxChannelIdleTimeout).To(gomega.BeNumerically(">", 0))
}

// Values that are durations still get through, zero included: it turns the
// release off rather than releasing at once.
func TestGetProfile_IdleTimeoutAcceptsSeconds(t *testing.T) {
	g := gomega.NewWithT(t)

	t.Setenv("MODAL_SANDBOX_CHANNEL_IDLE_TIMEOUT", "0")
	t.Setenv("MODAL_SANDBOX_STREAM_IDLE_TIMEOUT", "2.5")

	profile := getProfile("", config{})
	g.Expect(profile.SandboxChannelIdleTimeout).To(gomega.Equal(time.Duration(0)))
	g.Expect(profile.SandboxStreamIdleTimeout).To(gomega.Equal(2500 * time.Millisecond))
}
