package modal

import (
	"os"
	"path/filepath"
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
