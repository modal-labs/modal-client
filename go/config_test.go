package modal

import (
	"os"
	"path/filepath"
	"testing"

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

func TestProfileIsLocalhost(t *testing.T) {
	g := gomega.NewWithT(t)
	p := Profile{ServerURL: "http://localhost:8889"}
	g.Expect(p.isLocalhost()).Should(gomega.BeTrue())
}
