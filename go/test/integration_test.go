package test

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/onsi/gomega"
)

func TestIntegrationTests(t *testing.T) {
	g := gomega.NewWithT(t)

	cwd, err := os.Getwd()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	testScripts := []string{
		"integration/version-tracking/test.sh",
	}

	for _, relPath := range testScripts {
		path := filepath.Join(cwd, relPath)

		t.Run(relPath, func(t *testing.T) {
			g := gomega.NewWithT(t)

			_, err := os.Stat(path)
			g.Expect(err).ShouldNot(gomega.HaveOccurred(), "integration test script should exist at %s", path)

			cmd := exec.Command("sh", path)
			cmd.Dir = filepath.Dir(path)
			output, err := cmd.CombinedOutput()

			if err != nil {
				t.Logf("Integration test output:\n%s", string(output))
				t.Fatalf("Integration test failed: %v", err)
			}
		})
	}
}
