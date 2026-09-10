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

func TestGetProfile_OAuthCredentials(t *testing.T) {
	g := gomega.NewWithT(t)
	t.Setenv("MODAL_OAUTH_REFRESH_TOKEN", "refresh-token")
	t.Setenv("MODAL_OAUTH_CLIENT_ID", "oc-client-id")
	t.Setenv("MODAL_OAUTH_CLIENT_SECRET", "ov-client-secret")

	profile := getProfile("", config{})
	g.Expect(profile.OAuthRefreshToken).To(gomega.Equal("refresh-token"))
	g.Expect(profile.OAuthClientID).To(gomega.Equal("oc-client-id"))
	g.Expect(profile.OAuthClientSecret).To(gomega.Equal("ov-client-secret"))
}

func TestGetProfile_OAuthJWTKey(t *testing.T) {
	g := gomega.NewWithT(t)
	t.Setenv("MODAL_OAUTH_REFRESH_TOKEN", "refresh-token")
	t.Setenv("MODAL_OAUTH_CLIENT_ID", "oc-client-id")
	t.Setenv("MODAL_OAUTH_JWT_KEY", "private-key")

	profile := getProfile("", config{})
	g.Expect(profile.OAuthRefreshToken).To(gomega.Equal("refresh-token"))
	g.Expect(profile.OAuthClientID).To(gomega.Equal("oc-client-id"))
	g.Expect(profile.OAuthJWTKey).To(gomega.Equal("private-key"))
	g.Expect(profile.OAuthClientSecret).To(gomega.BeEmpty())
}

func TestGetProfile_OAuthJWTEnvClearsFileSecret(t *testing.T) {
	g := gomega.NewWithT(t)
	t.Setenv("MODAL_TOKEN_ID", "")
	t.Setenv("MODAL_TOKEN_SECRET", "")
	t.Setenv("MODAL_OAUTH_CLIENT_SECRET", "")
	t.Setenv("MODAL_OAUTH_JWT_KEY", "env-jwt-key")

	profile := getProfile("oauth-profile", config{
		"oauth-profile": rawProfile{
			OAuthRefreshToken: "refresh-token",
			OAuthClientID:     "oc-client-id",
			OAuthClientSecret: "ov-file-secret",
		},
	})
	g.Expect(profile.OAuthJWTKey).To(gomega.Equal("env-jwt-key"))
	g.Expect(profile.OAuthClientSecret).To(gomega.BeEmpty())
	g.Expect(validateProfileCredentials(profile, false)).ShouldNot(gomega.HaveOccurred())
}

func TestGetProfile_OAuthSecretEnvClearsFileJWT(t *testing.T) {
	g := gomega.NewWithT(t)
	t.Setenv("MODAL_TOKEN_ID", "")
	t.Setenv("MODAL_TOKEN_SECRET", "")
	t.Setenv("MODAL_OAUTH_CLIENT_SECRET", "ov-env-secret")
	t.Setenv("MODAL_OAUTH_JWT_KEY", "")

	profile := getProfile("oauth-profile", config{
		"oauth-profile": rawProfile{
			OAuthRefreshToken: "refresh-token",
			OAuthClientID:     "oc-client-id",
			OAuthJWTKey:       "file-jwt-key",
		},
	})
	g.Expect(profile.OAuthClientSecret).To(gomega.Equal("ov-env-secret"))
	g.Expect(profile.OAuthJWTKey).To(gomega.BeEmpty())
	g.Expect(validateProfileCredentials(profile, false)).ShouldNot(gomega.HaveOccurred())
}

func TestGetProfile_OAuthBothEnvMethodsKept(t *testing.T) {
	g := gomega.NewWithT(t)
	t.Setenv("MODAL_OAUTH_CLIENT_SECRET", "ov-env-secret")
	t.Setenv("MODAL_OAUTH_JWT_KEY", "env-jwt-key")

	profile := getProfile("", config{})
	g.Expect(profile.OAuthClientSecret).To(gomega.Equal("ov-env-secret"))
	g.Expect(profile.OAuthJWTKey).To(gomega.Equal("env-jwt-key"))
}

func TestGetProfile_OAuthCredentialsFromConfig(t *testing.T) {
	g := gomega.NewWithT(t)
	t.Setenv("MODAL_OAUTH_REFRESH_TOKEN", "")
	t.Setenv("MODAL_OAUTH_CLIENT_ID", "")
	t.Setenv("MODAL_OAUTH_CLIENT_SECRET", "")
	t.Setenv("MODAL_OAUTH_JWT_KEY", "")

	profile := getProfile("oauth-profile", config{
		"oauth-profile": rawProfile{
			OAuthRefreshToken: "refresh-token",
			OAuthClientID:     "oc-client-id",
			OAuthClientSecret: "ov-client-secret",
			OAuthJWTKey:       "private-key",
		},
	})
	g.Expect(profile.OAuthRefreshToken).To(gomega.Equal("refresh-token"))
	g.Expect(profile.OAuthClientID).To(gomega.Equal("oc-client-id"))
	g.Expect(profile.OAuthClientSecret).To(gomega.Equal("ov-client-secret"))
	g.Expect(profile.OAuthJWTKey).To(gomega.Equal("private-key"))
}

func TestGetProfile_SandboxV2Parsing(t *testing.T) {
	cases := []struct {
		envVal   string
		expected bool
	}{
		{"", false},
		{"0", false},
		{"false", false},
		{"False", false},
		{"1", true},
		{"true", true},
		{"yes", true},
	}

	for _, tc := range cases {
		t.Run("MODAL_SANDBOX_V2="+tc.envVal, func(t *testing.T) {
			g := gomega.NewWithT(t)
			t.Setenv("MODAL_SANDBOX_V2", tc.envVal)
			profile := getProfile("", config{})
			g.Expect(profile.SandboxV2).To(gomega.Equal(tc.expected))
		})
	}
}

func TestGetProfile_SandboxV2FromConfigFile(t *testing.T) {
	g := gomega.NewWithT(t)
	t.Setenv("MODAL_SANDBOX_V2", "")

	profile := getProfile("v2-profile", config{"v2-profile": rawProfile{SandboxV2: true}})
	g.Expect(profile.SandboxV2).To(gomega.BeTrue())
}

func TestGetProfile_SandboxV2EnvOverridesConfigFile(t *testing.T) {
	g := gomega.NewWithT(t)
	t.Setenv("MODAL_SANDBOX_V2", "0")

	profile := getProfile("v2-profile", config{"v2-profile": rawProfile{SandboxV2: true}})
	g.Expect(profile.SandboxV2).To(gomega.BeFalse())
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
	}
	bad := []string{"Inf", "+Inf", "Infinity", "-Inf", "NaN", "1e30", "-1", "nonsense"}

	for _, tc := range cases {
		for _, value := range bad {
			t.Run(tc.envVar+"="+value, func(t *testing.T) {
				g := gomega.NewWithT(t)
				t.Setenv(tc.envVar, value)

				profile := getProfile("", config{})
				got := profile.SandboxChannelIdleTimeout
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

	profile := getProfile("", config{})
	g.Expect(profile.SandboxChannelIdleTimeout).To(gomega.BeNumerically(">", 0))
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

	profile := getProfile("", config{})
	g.Expect(profile.SandboxChannelIdleTimeout).To(gomega.Equal(time.Duration(0)))
}
