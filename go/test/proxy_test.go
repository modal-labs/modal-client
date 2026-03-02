package test

import (
	"strings"
	"testing"

	modal "github.com/modal-labs/modal-client/go"
	"github.com/onsi/gomega"
)

func TestCreateSandboxWithProxy(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	proxy, err := tc.Proxies.FromName(ctx, "libmodal-test-proxy", &modal.ProxyFromNameParams{Environment: "libmodal"})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(strings.HasPrefix(proxy.ProxyID, "pr-")).To(gomega.BeTrue())

	sb, err := tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Proxy:   proxy,
		Command: []string{"echo", "hello, Sandbox with Proxy"},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)
	g.Expect(sb.SandboxID).ShouldNot(gomega.BeEmpty())
}

func TestProxyNotFound(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	_, err := tc.Proxies.FromName(ctx, "non-existent-proxy-name", nil)
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("Proxy 'non-existent-proxy-name' not found"))
}
