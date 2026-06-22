package test

import (
	"fmt"
	"io"
	"math/rand"
	"strings"
	"testing"
	"time"

	modal "github.com/modal-labs/modal-client/go"
	"github.com/modal-labs/modal-client/go/internal/grpcmock"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"github.com/onsi/gomega"
)

const (
	validV1SandboxID = "sb-nGEijt9WbBMlGrsPH9FOaC"
	validV2SandboxID = "sb-01ARZ3NDEKTSV4RRFFQ69G5FAV"
)

func TestCreateOneSandbox(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(sb.SandboxID).ShouldNot(gomega.BeEmpty())

	exitcode, err := sb.Terminate(ctx, &modal.SandboxTerminateParams{Wait: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(exitcode).To(gomega.Equal(137))
}

func TestCreateOneSandboxTerminateWaitWorks(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(sb.SandboxID).ShouldNot(gomega.BeEmpty())

	_, err = sb.Terminate(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	exitcode, err := sb.Wait(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(exitcode).To(gomega.Equal(137))
}

func TestPassCatToStdin(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{Command: []string{"cat"}})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	_, err = sb.Stdin.Write([]byte("this is input that should be mirrored by cat"))
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	err = sb.Stdin.Close()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	output, err := io.ReadAll(sb.Stdout)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(string(output)).To(gomega.Equal("this is input that should be mirrored by cat"))
}

func TestIgnoreLargeStdout(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("python:3.13-alpine", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	p, err := sb.Exec(ctx, []string{"python", "-c", `print("a" * 1_000_000)`}, &modal.SandboxExecParams{Stdout: modal.Ignore})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	buf, err := io.ReadAll(p.Stdout)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(len(buf)).To(gomega.Equal(0)) // Stdout is ignored

	// Stdout should be consumed after cancel, without blocking the process.
	exitCode, err := p.Wait(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(exitCode).To(gomega.Equal(0))
}

func TestSandboxExecWaitSignal(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ToNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	// The shell kills itself with SIGKILL (9); wait() should return 128 + 9 = 137.
	p, err := sb.Exec(ctx, []string{"sh", "-c", "kill -9 $$"}, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	exitCode, err := p.Wait(ctx, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(exitCode).To(gomega.Equal(128 + 9))
}

func TestSandboxCreateOptions(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{
		CreateIfMissing: true,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Command: []string{"echo", "hello, params"},
		Cloud:   "aws",
		Regions: []string{"us-east-1", "us-west-2"},
		Verbose: true,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)
	g.Expect(sb.SandboxID).Should(gomega.HavePrefix("sb-"))

	exitCode, err := sb.Wait(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(exitCode).Should(gomega.Equal(0))

	_, err = tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Cloud: "invalid-cloud",
	})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).Should(gomega.ContainSubstring("InvalidArgument"))

	_, err = tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Regions: []string{"invalid-region"},
	})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).Should(gomega.ContainSubstring("InvalidArgument"))
}

func TestSandboxExecOptions(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	p, err := sb.Exec(ctx, []string{"pwd"}, &modal.SandboxExecParams{
		Workdir: "/tmp",
		Timeout: 5 * time.Second,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	output, err := io.ReadAll(p.Stdout)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(string(output)).To(gomega.Equal("/tmp\n"))

	exitCode, err := p.Wait(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(exitCode).To(gomega.Equal(0))
}

func TestSandboxWithVolume(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{
		CreateIfMissing: true,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	volume, err := tc.Volumes.FromName(ctx, "libmodal-test-sandbox-volume", &modal.VolumeFromNameParams{
		CreateIfMissing: true,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	sb, err := tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Command: []string{"echo", "volume test"},
		Volumes: map[string]*modal.Volume{
			"/mnt/test": volume,
		},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	exitCode, err := sb.Wait(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(exitCode).Should(gomega.Equal(0))
}

func TestSandboxWithReadOnlyVolume(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{
		CreateIfMissing: true,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	volume, err := tc.Volumes.FromName(ctx, "libmodal-test-sandbox-volume", &modal.VolumeFromNameParams{
		CreateIfMissing: true,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	trueVal := true
	readOnlyVolume := volume.WithMountOptions(&modal.VolumeMountOptions{ReadOnly: &trueVal})

	sb, err := tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Command: []string{"sh", "-c", "echo 'test' > /mnt/test/test.txt"},
		Volumes: map[string]*modal.Volume{
			"/mnt/test": readOnlyVolume,
		},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	exitCode, err := sb.Wait(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(exitCode).Should(gomega.Equal(1))

	stderr, err := io.ReadAll(sb.Stderr)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(string(stderr)).Should(gomega.ContainSubstring("Read-only file system"))
}

func TestSandboxWithSubPathVolume(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{
		CreateIfMissing: true,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	volume, err := tc.Volumes.Ephemeral(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer volume.CloseEphemeral()

	subPath := "/scoped"
	subPathVolume := volume.WithMountOptions(&modal.VolumeMountOptions{SubPath: &subPath})

	// Write a marker file into the sub-path-mounted volume.
	writer, err := tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Command: []string{"sh", "-c", "echo subpath-works > /mnt/sub/marker.txt"},
		Volumes: map[string]*modal.Volume{
			"/mnt/sub": subPathVolume,
		},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, writer)

	exitCode, err := writer.Wait(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(exitCode).Should(gomega.Equal(0))

	// Mount the same volume at the root and verify the file landed under the sub-path.
	reader, err := tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Command: []string{"cat", "/mnt/full/scoped/marker.txt"},
		Volumes: map[string]*modal.Volume{
			"/mnt/full": volume,
		},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, reader)

	exitCode, err = reader.Wait(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(exitCode).Should(gomega.Equal(0))

	stdout, err := io.ReadAll(reader.Stdout)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(strings.TrimSpace(string(stdout))).Should(gomega.Equal("subpath-works"))
}

func TestSandboxWithTunnels(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{
		CreateIfMissing: true,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Command:          []string{"cat"},
		EncryptedPorts:   []int{8443},
		UnencryptedPorts: []int{8080},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	g.Expect(sb.SandboxID).Should(gomega.HavePrefix("sb-"))

	tunnels, err := sb.Tunnels(ctx, 30*time.Second, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	g.Expect(tunnels).Should(gomega.HaveLen(2))

	// Test encrypted tunnel (port 8443)
	encryptedTunnel := tunnels[8443]
	g.Expect(encryptedTunnel.Host).Should(gomega.MatchRegexp(`\.modal\.host$`))
	g.Expect(encryptedTunnel.Port).Should(gomega.Equal(443))
	g.Expect(encryptedTunnel.URL()).Should(gomega.HavePrefix("https://"))

	host, port := encryptedTunnel.TLSSocket()
	g.Expect(host).Should(gomega.Equal(encryptedTunnel.Host))
	g.Expect(port).Should(gomega.Equal(encryptedTunnel.Port))

	// Test unencrypted tunnel (port 8080)
	unencryptedTunnel := tunnels[8080]
	g.Expect(unencryptedTunnel.UnencryptedHost).Should(gomega.MatchRegexp(`\.modal\.host$`))
	g.Expect(unencryptedTunnel.UnencryptedPort).Should(gomega.BeNumerically(">", 0))

	tcpHost, tcpPort, err := unencryptedTunnel.TCPSocket()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(tcpHost).Should(gomega.Equal(unencryptedTunnel.UnencryptedHost))
	g.Expect(tcpPort).Should(gomega.Equal(unencryptedTunnel.UnencryptedPort))
}

func TestCreateSandboxWithSecrets(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	secret, err := tc.Secrets.FromName(ctx, "libmodal-test-secret", &modal.SecretFromNameParams{RequiredKeys: []string{"c"}})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{Secrets: []*modal.Secret{secret}, Command: []string{"printenv", "c"}})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	output, err := io.ReadAll(sb.Stdout)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(string(output)).To(gomega.Equal("hello world\n"))
}

func TestSandboxPollAndReturnCode(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{Command: []string{"cat"}})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	pollResult, err := sb.Poll(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(pollResult).Should(gomega.BeNil())

	// Send input to make the cat command complete
	_, err = sb.Stdin.Write([]byte("hello, Sandbox"))
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	err = sb.Stdin.Close()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	waitResult, err := sb.Wait(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(waitResult).To(gomega.Equal(0))

	pollResult, err = sb.Poll(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(pollResult).ShouldNot(gomega.BeNil())
	g.Expect(*pollResult).To(gomega.Equal(0))
}

func TestSandboxPollAfterFailure(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Command: []string{"sh", "-c", "exit 42"},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	waitResult, err := sb.Wait(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(waitResult).To(gomega.Equal(42))

	pollResult, err := sb.Poll(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(pollResult).ShouldNot(gomega.BeNil())
	g.Expect(*pollResult).To(gomega.Equal(42))
}

func TestCreateSandboxWithNetworkAccessParams(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{
		CreateIfMissing: true,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Command:               []string{"echo", "hello, network access"},
		BlockNetwork:          false,
		OutboundCIDRAllowlist: []string{"10.0.0.0/8", "192.168.0.0/16"},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	g.Expect(sb).ShouldNot(gomega.BeNil())
	g.Expect(sb.SandboxID).Should(gomega.HavePrefix("sb-"))

	exitCode, err := sb.Wait(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(exitCode).Should(gomega.Equal(0))

	_, err = tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		BlockNetwork:          false,
		OutboundCIDRAllowlist: []string{"not-an-ip/8"},
	})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).Should(gomega.ContainSubstring("Invalid CIDR: not-an-ip/8"))

	_, err = tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		BlockNetwork:          true,
		OutboundCIDRAllowlist: []string{"10.0.0.0/8"},
	})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).Should(gomega.ContainSubstring("OutboundCIDRAllowlist cannot be used when BlockNetwork is enabled"))

}

func TestCreateSandboxWithDomainAllowlist(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{
		CreateIfMissing: true,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	// Domain-only allowlist.
	sb, err := tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Command:                 []string{"echo", "hello, domain allowlist"},
		OutboundDomainAllowlist: []string{"example.com", "*.github.com"},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	g.Expect(sb).ShouldNot(gomega.BeNil())
	g.Expect(sb.SandboxID).Should(gomega.HavePrefix("sb-"))

	exitCode, err := sb.Wait(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(exitCode).Should(gomega.Equal(0))

	// Domain + CIDR combined.
	sb2, err := tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Command:                 []string{"echo", "hello, domain+cidr"},
		OutboundDomainAllowlist: []string{"api.example.com"},
		OutboundCIDRAllowlist:   []string{"10.0.0.0/8"},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb2)

	exitCode2, err := sb2.Wait(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(exitCode2).Should(gomega.Equal(0))

	// Cannot combine with BlockNetwork.
	_, err = tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		BlockNetwork:            true,
		OutboundDomainAllowlist: []string{"example.com"},
	})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).Should(gomega.ContainSubstring("OutboundDomainAllowlist cannot be used when BlockNetwork is enabled"))

	// Invalid domain triggers server-side validation error.
	_, err = tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		OutboundDomainAllowlist: []string{"not a valid domain!"},
	})
	g.Expect(err).Should(gomega.HaveOccurred())
}

func TestSandboxExecSecret(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	secret, err := tc.Secrets.FromName(ctx, "libmodal-test-secret", &modal.SecretFromNameParams{RequiredKeys: []string{"c"}})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	secret2, err := tc.Secrets.FromMap(ctx, map[string]string{"d": "3"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	p, err := sb.Exec(ctx, []string{"printenv", "c", "d"}, &modal.SandboxExecParams{Secrets: []*modal.Secret{secret, secret2}})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	buf, err := io.ReadAll(p.Stdout)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(string(buf)).Should(gomega.Equal("hello world\n3\n"))
}

func TestSandboxModalIdentityTokenUnsetByDefault(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Command: []string{"sh", "-c", "echo ${MODAL_IDENTITY_TOKEN:-UNSET}"},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	output, err := io.ReadAll(sb.Stdout)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(strings.TrimSpace(string(output))).To(gomega.Equal("UNSET"))

	exitCode, err := sb.Wait(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(exitCode).To(gomega.Equal(0))
}

func TestSandboxIncludeOidcIdentityTokenSetsModalIdentityTokenEnv(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Command:                  []string{"sh", "-c", "echo ${MODAL_IDENTITY_TOKEN:-UNSET}"},
		IncludeOidcIdentityToken: true,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	output, err := io.ReadAll(sb.Stdout)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	token := strings.TrimSpace(string(output))
	g.Expect(token).NotTo(gomega.Equal("UNSET"))
	g.Expect(token).NotTo(gomega.BeEmpty())

	exitCode, err := sb.Wait(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(exitCode).To(gomega.Equal(0))
}

func TestSandboxFromId(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	g.Expect(sb.SandboxID).ShouldNot(gomega.BeEmpty())

	sbFromID, err := tc.Sandboxes.FromID(ctx, sb.SandboxID, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sbFromID)

	g.Expect(sbFromID.SandboxID).Should(gomega.Equal(sb.SandboxID))
}

func TestSandboxWithWorkdir(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Command: []string{"pwd"},
		Workdir: "/tmp",
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	output, err := io.ReadAll(sb.Stdout)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(string(output)).To(gomega.Equal("/tmp\n"))

	exitCode, err := sb.Wait(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(exitCode).To(gomega.Equal(0))

	_, err = tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Workdir: "relative/path",
	})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("workdir must be an absolute path"))
}

func TestSandboxSetTagsAndList(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	unique := fmt.Sprintf("%d", rand.Int())

	var before []string
	it, err := tc.Sandboxes.List(ctx, &modal.SandboxListParams{Tags: map[string]string{"test-key": unique}})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	for s, err := range it {
		g.Expect(err).ShouldNot(gomega.HaveOccurred())
		before = append(before, s.SandboxID)
	}
	g.Expect(before).To(gomega.HaveLen(0))

	err = sb.SetTags(ctx, map[string]string{"test-key": unique}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	var after []string
	it, err = tc.Sandboxes.List(ctx, &modal.SandboxListParams{Tags: map[string]string{"test-key": unique}})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	for s, err := range it {
		g.Expect(err).ShouldNot(gomega.HaveOccurred())
		after = append(after, s.SandboxID)
	}
	g.Expect(after).To(gomega.Equal([]string{sb.SandboxID}))
}

func TestSandboxTags(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	retrievedTagsBefore, err := sb.GetTags(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(retrievedTagsBefore).To(gomega.Equal(map[string]string{}))

	tagA := fmt.Sprintf("%d", rand.Int())
	tagB := fmt.Sprintf("%d", rand.Int())
	tagC := fmt.Sprintf("%d", rand.Int())

	err = sb.SetTags(ctx, map[string]string{"key-a": tagA, "key-b": tagB, "key-c": tagC}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	retrievedTags, err := sb.GetTags(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(retrievedTags).To(gomega.Equal(map[string]string{"key-a": tagA, "key-b": tagB, "key-c": tagC}))

	var ids []string
	it, err := tc.Sandboxes.List(ctx, &modal.SandboxListParams{Tags: map[string]string{"key-a": tagA}})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	for s, err := range it {
		g.Expect(err).ShouldNot(gomega.HaveOccurred())
		ids = append(ids, s.SandboxID)
	}
	g.Expect(ids).To(gomega.Equal([]string{sb.SandboxID}))

	ids = nil
	it, err = tc.Sandboxes.List(ctx, &modal.SandboxListParams{Tags: map[string]string{"key-a": tagA, "key-b": tagB}})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	for s, err := range it {
		g.Expect(err).ShouldNot(gomega.HaveOccurred())
		ids = append(ids, s.SandboxID)
	}
	g.Expect(ids).To(gomega.Equal([]string{sb.SandboxID}))

	ids = nil
	it, err = tc.Sandboxes.List(ctx, &modal.SandboxListParams{Tags: map[string]string{"key-a": tagA, "key-b": tagB, "key-d": "not-set"}})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	for s, err := range it {
		g.Expect(err).ShouldNot(gomega.HaveOccurred())
		ids = append(ids, s.SandboxID)
	}
	g.Expect(ids).To(gomega.HaveLen(0))
}

func TestSandboxListByAppId(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	count := 0
	it, err := tc.Sandboxes.List(ctx, &modal.SandboxListParams{AppID: app.AppID})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	for s, err := range it {
		g.Expect(err).ShouldNot(gomega.HaveOccurred())
		g.Expect(s.SandboxID).Should(gomega.HavePrefix("sb-"))
		count++
		if count >= 1 {
			break
		}
	}
	g.Expect(count).ToNot(gomega.Equal(0))
}

func TestNamedSandbox(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sandboxName := fmt.Sprintf("test-sandbox-%d", rand.Int())

	sb, err := tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Name:    sandboxName,
		Command: []string{"sleep", "60"},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)
	g.Expect(sb.SandboxID).ShouldNot(gomega.BeEmpty())

	sb1FromName, err := tc.Sandboxes.FromName(ctx, "libmodal-test", sandboxName, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(sb1FromName.SandboxID).To(gomega.Equal(sb.SandboxID))

	sb2FromName, err := tc.Sandboxes.FromName(ctx, "libmodal-test", sandboxName, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(sb2FromName.SandboxID).To(gomega.Equal(sb1FromName.SandboxID))

	_, err = tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Name:    sandboxName,
		Command: []string{"sleep", "60"},
	})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("already exists"))
}

func TestNamedSandboxNotFound(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	_, err := tc.Sandboxes.FromName(ctx, "libmodal-test", "non-existent-sandbox", nil)
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("not found"))
}

func TestConnectToken(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("python:3.12-alpine", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	creds, err := sb.CreateConnectToken(ctx, &modal.SandboxCreateConnectTokenParams{UserMetadata: "abc"})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(creds.Token).ShouldNot(gomega.BeEmpty())
	g.Expect(creds.URL).ShouldNot(gomega.BeEmpty())
}

func TestCreateConnectTokenSendsPort(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(mock, "SandboxCreateConnectToken",
		func(req *pb.SandboxCreateConnectTokenRequest) (*pb.SandboxCreateConnectTokenResponse, error) {
			g.Expect(req.GetSandboxId()).To(gomega.Equal(validV1SandboxID))
			g.Expect(req.HasPort()).To(gomega.BeTrue())
			g.Expect(req.GetPort()).To(gomega.Equal(uint32(9000)))
			return pb.SandboxCreateConnectTokenResponse_builder{
				Token: "token-9000",
			}.Build(), nil
		})

	sb, err := mock.Sandboxes.FromID(ctx, validV1SandboxID, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	creds, err := sb.CreateConnectToken(ctx, &modal.SandboxCreateConnectTokenParams{Port: 9000})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(creds.Token).To(gomega.Equal("token-9000"))

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestCreateConnectTokenDefaultsTo8080(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(mock, "SandboxCreateConnectToken",
		func(req *pb.SandboxCreateConnectTokenRequest) (*pb.SandboxCreateConnectTokenResponse, error) {
			g.Expect(req.HasPort()).To(gomega.BeTrue())
			g.Expect(req.GetPort()).To(gomega.Equal(uint32(8080)))
			return pb.SandboxCreateConnectTokenResponse_builder{
				Token: "token",
			}.Build(), nil
		})

	sb, err := mock.Sandboxes.FromID(ctx, validV1SandboxID, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	creds, err := sb.CreateConnectToken(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(creds.Token).To(gomega.Equal("token"))

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestCreateConnectTokenRejectsInvalidPort(t *testing.T) {
	t.Parallel()

	for _, port := range []int{-1, 65536} {
		t.Run(fmt.Sprintf("port=%d", port), func(t *testing.T) {
			t.Parallel()
			g := gomega.NewWithT(t)
			ctx := t.Context()
			mock := newGRPCMockClient(t)

			sb, err := mock.Sandboxes.FromID(ctx, validV1SandboxID, nil)
			g.Expect(err).ShouldNot(gomega.HaveOccurred())

			_, err = sb.CreateConnectToken(ctx, &modal.SandboxCreateConnectTokenParams{Port: port})
			g.Expect(err).Should(gomega.HaveOccurred())
			g.Expect(err.Error()).To(gomega.ContainSubstring("expects Port in [1, 65535]"))

			g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
		})
	}
}

func TestSandboxInvalidTimeouts(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	_, err = tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{Timeout: -1 * time.Second})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).Should(gomega.ContainSubstring("must be non-negative"))

	_, err = tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{Timeout: 1500 * time.Millisecond})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).Should(gomega.ContainSubstring("whole number of seconds"))

	_, err = tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{IdleTimeout: -2 * time.Second})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).Should(gomega.ContainSubstring("must be non-negative"))

	_, err = tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{IdleTimeout: 2500 * time.Millisecond})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).Should(gomega.ContainSubstring("whole number of seconds"))

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	_, err = sb.Exec(ctx, []string{"echo", "test"}, &modal.SandboxExecParams{Timeout: -5 * time.Second})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).Should(gomega.ContainSubstring("must be non-negative"))

	_, err = sb.Exec(ctx, []string{"echo", "test"}, &modal.SandboxExecParams{Timeout: 3500 * time.Millisecond})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).Should(gomega.ContainSubstring("whole number of seconds"))
}

func TestSandboxExperimentalDocker(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	// With experimental option should include /var/lib/docker
	options := map[string]any{"enable_docker": true}
	sb, err := tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{ExperimentalOptions: options})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	p, err := sb.Exec(ctx, []string{"test", "-d", "/var/lib/docker"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	exitCode, err := p.Wait(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(exitCode).Should(gomega.Equal(0))

	// Without experimental option should **not** include /var/lib/docker
	sbDefault, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sbDefault)
	p, err = sbDefault.Exec(ctx, []string{"test", "-d", "/var/lib/docker"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	exitCode, err = p.Wait(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(exitCode).Should(gomega.Equal(1))
}

func TestSandboxExperimentalDockerNotBool(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	options := map[string]any{"enable_docker": "not-a-bool"}
	_, err = tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{ExperimentalOptions: options})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).Should(gomega.ContainSubstring("must be a bool"))
}

func TestSandboxExperimentalDockerMock(t *testing.T) {
	// Unset `MODAL_IMAGE_BUILDER_VERSION` so that this test uses the image builder version from `EnvironmentGetOrCreate`.
	t.Setenv("MODAL_IMAGE_BUILDER_VERSION", "")
	g := gomega.NewWithT(t)

	options := map[string]any{"enable_docker": true}
	expectedOptoins := map[string]bool{"enable_docker": true}
	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(
		mock, "SandboxCreate",
		func(req *pb.SandboxCreateRequest) (*pb.SandboxCreateResponse, error) {
			g.Expect(req.GetDefinition().GetExperimentalOptions()).Should(gomega.Equal(expectedOptoins)) //nolint:staticcheck // testing deprecated field
			return pb.SandboxCreateResponse_builder{
				SandboxId: validV1SandboxID,
			}.Build(), nil
		},
	)
	grpcmock.HandleUnary(
		mock, "AppGetOrCreate",
		func(req *pb.AppGetOrCreateRequest) (*pb.AppGetOrCreateResponse, error) {
			return pb.AppGetOrCreateResponse_builder{
				AppId: "ap-1234",
			}.Build(), nil
		},
	)

	grpcmock.HandleUnary(
		mock, "ImageGetOrCreate",
		func(req *pb.ImageGetOrCreateRequest) (*pb.ImageGetOrCreateResponse, error) {
			return pb.ImageGetOrCreateResponse_builder{
				ImageId: "im-123",
				Result: pb.GenericResult_builder{
					Status: pb.GenericResult_GENERIC_STATUS_SUCCESS,
				}.Build(),
			}.Build(), nil
		},
	)
	grpcmock.HandleUnary(
		mock, "EnvironmentGetOrCreate",
		func(req *pb.EnvironmentGetOrCreateRequest) (*pb.EnvironmentGetOrCreateResponse, error) {
			return pb.EnvironmentGetOrCreateResponse_builder{
				EnvironmentId: "en-test",
				Metadata: pb.EnvironmentMetadata_builder{
					Name: "test",
					Settings: pb.EnvironmentSettings_builder{
						ImageBuilderVersion: "2025.06",
					}.Build(),
				}.Build(),
			}.Build(), nil
		},
	)

	ctx := t.Context()
	app, err := mock.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := mock.Images.FromRegistry("alpine:3.21", nil)
	sb, err := mock.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{ExperimentalOptions: options})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	g.Expect(sb.SandboxID).Should(gomega.Equal(validV1SandboxID))

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestSandboxGetTaskIdPolling(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(mock, "SandboxGetTaskId",
		func(req *pb.SandboxGetTaskIdRequest) (*pb.SandboxGetTaskIdResponse, error) {
			return pb.SandboxGetTaskIdResponse_builder{}.Build(), nil
		})
	grpcmock.HandleUnary(mock, "SandboxGetTaskId",
		func(req *pb.SandboxGetTaskIdRequest) (*pb.SandboxGetTaskIdResponse, error) {
			taskID := "ta-123"
			return pb.SandboxGetTaskIdResponse_builder{
				TaskId: &taskID,
			}.Build(), nil
		})

	sb, err := mock.Sandboxes.FromID(ctx, validV1SandboxID, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	_, err = sb.Exec(ctx, []string{"echo", "hello"}, nil)
	g.Expect(err).Should(gomega.HaveOccurred())

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestSandboxGetTaskIdTerminated(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	mock := newGRPCMockClient(t)

	grpcmock.HandleUnary(mock, "SandboxGetTaskId",
		func(req *pb.SandboxGetTaskIdRequest) (*pb.SandboxGetTaskIdResponse, error) {
			return pb.SandboxGetTaskIdResponse_builder{
				TaskResult: pb.GenericResult_builder{
					Status: pb.GenericResult_GENERIC_STATUS_TERMINATED,
				}.Build(),
			}.Build(), nil
		})

	sb, err := mock.Sandboxes.FromID(ctx, validV1SandboxID, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	_, err = sb.Exec(ctx, []string{"echo", "hello"}, nil)
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).Should(gomega.ContainSubstring("already completed"))

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestSandboxFromIDRoutesV1(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	mock := newGRPCMockClient(t)
	sandboxID := validV1SandboxID

	grpcmock.HandleUnary(mock, "SandboxWait",
		func(req *pb.SandboxWaitRequest) (*pb.SandboxWaitResponse, error) {
			g.Expect(req.GetSandboxId()).To(gomega.Equal(sandboxID))
			g.Expect(req.GetTimeout()).To(gomega.Equal(float32(0)))
			return pb.SandboxWaitResponse_builder{
				Result: pb.GenericResult_builder{
					Status:   pb.GenericResult_GENERIC_STATUS_SUCCESS,
					Exitcode: 0,
				}.Build(),
			}.Build(), nil
		})

	sb, err := mock.Sandboxes.FromID(ctx, sandboxID, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	exitCode, err := sb.Poll(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(exitCode).ShouldNot(gomega.BeNil())
	g.Expect(*exitCode).To(gomega.Equal(0))

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestSandboxFromIDRoutesV2(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	mock := newGRPCMockClient(t)
	sandboxID := validV2SandboxID

	grpcmock.HandleUnary(mock, "SandboxTerminateV2",
		func(req *pb.SandboxTerminateRequest) (*pb.SandboxTerminateResponse, error) {
			g.Expect(req.GetSandboxId()).To(gomega.Equal(sandboxID))
			return pb.SandboxTerminateResponse_builder{}.Build(), nil
		})
	grpcmock.HandleUnary(mock, "SandboxWaitV2",
		func(req *pb.SandboxWaitRequest) (*pb.SandboxWaitResponse, error) {
			g.Expect(req.GetSandboxId()).To(gomega.Equal(sandboxID))
			g.Expect(req.GetTimeout()).To(gomega.Equal(float32(10)))
			return pb.SandboxWaitResponse_builder{
				Result: pb.GenericResult_builder{
					Status: pb.GenericResult_GENERIC_STATUS_TERMINATED,
				}.Build(),
			}.Build(), nil
		})

	sb, err := mock.Sandboxes.FromID(ctx, sandboxID, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	exitCode, err := sb.Terminate(ctx, &modal.SandboxTerminateParams{Wait: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(exitCode).To(gomega.Equal(137))

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestSandboxFromIDRejectsInvalidID(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	mock := newGRPCMockClient(t)

	_, err := mock.Sandboxes.FromID(ctx, "sb-123", nil)
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("Invalid Sandbox ID"))

	g.Expect(mock.AssertExhausted()).ShouldNot(gomega.HaveOccurred())
}

func TestSandboxDetachIsNonDestructive(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ToNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())

	sandboxID := sb.SandboxID

	err = sb.Detach()
	g.Expect(err).ToNot(gomega.HaveOccurred())

	sbFromID, err := tc.Sandboxes.FromID(ctx, sandboxID, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sbFromID)
	g.Expect(sbFromID.SandboxID).To(gomega.Equal(sandboxID))

	p, err := sbFromID.Exec(ctx, []string{"echo", "still running"}, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	exitCode, err := p.Wait(ctx, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(exitCode).To(gomega.Equal(0))
}

func TestSandboxDetachIsIdempotent(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ToNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	sbFromID, err := tc.Sandboxes.FromID(ctx, sb.SandboxID, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sbFromID)

	err = sb.Detach()
	g.Expect(err).ToNot(gomega.HaveOccurred())
	err = sb.Detach()
	g.Expect(err).ToNot(gomega.HaveOccurred())
	err = sb.Detach()
	g.Expect(err).ToNot(gomega.HaveOccurred())
}

func TestSandboxTerminateThenDetachDoesNotError(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ToNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())

	_, err = sb.Terminate(ctx, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())

	err = sb.Detach()
	g.Expect(err).ToNot(gomega.HaveOccurred())
}

func TestSandboxDetachForbidsAllOperations(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ToNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	sbFromID, err := tc.Sandboxes.FromID(ctx, sb.SandboxID, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sbFromID)

	err = sb.Detach()
	g.Expect(err).ToNot(gomega.HaveOccurred())

	errorMsg := "Unable to perform operation on a detached sandbox"

	_, err = sb.Exec(ctx, []string{"echo", "hello"}, nil)
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring(errorMsg))

	_, err = sb.CreateConnectToken(ctx, nil)
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring(errorMsg))

	_, err = sb.Tunnels(ctx, 30*time.Second, nil)
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring(errorMsg))

	_, err = sb.SnapshotFilesystem(ctx, &modal.SandboxSnapshotFilesystemParams{Timeout: 30 * time.Second})
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring(errorMsg))

	err = sb.MountImage(ctx, "/abc", nil, nil)
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring(errorMsg))

	_, err = sb.SnapshotDirectory(ctx, "/abc", nil)
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring(errorMsg))

	_, err = sb.Poll(ctx, nil)
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring(errorMsg))

	err = sb.SetTags(ctx, map[string]string{}, nil)
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring(errorMsg))

	_, err = sb.GetTags(ctx, nil)
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring(errorMsg))

	err = sb.WaitUntilReady(ctx, 1*time.Second, nil)
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring(errorMsg))
}

func TestSandboxExecStdinStdout(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	p, err := sb.Exec(ctx, []string{"sh", "-c", "while read line; do echo $line; done"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	_, err = p.Stdin.Write([]byte("foo\n"))
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	_, err = p.Stdin.Write([]byte("bar\n"))
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	err = p.Stdin.Close()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	output, err := io.ReadAll(p.Stdout)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(string(output)).To(gomega.Equal("foo\nbar\n"))
}

func TestSandboxExecWaitExitCode(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	p, err := sb.Exec(ctx, []string{"sh", "-c", "exit 42"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	exitCode, err := p.Wait(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(exitCode).To(gomega.Equal(42))
}

func TestSandboxExecDoubleRead(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	p, err := sb.Exec(ctx, []string{"echo", "hello"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	output1, err := io.ReadAll(p.Stdout)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(string(output1)).To(gomega.Equal("hello\n"))

	output2, err := io.ReadAll(p.Stdout)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(string(output2)).To(gomega.Equal(""))

	exitCode, err := p.Wait(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(exitCode).To(gomega.Equal(0))
}

func TestSandboxExecBinaryMode(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	p, err := sb.Exec(ctx, []string{"printf", "\\x01\\x02\\x03"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	output, err := io.ReadAll(p.Stdout)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(output).To(gomega.Equal([]byte{0x01, 0x02, 0x03}))

	exitCode, err := p.Wait(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(exitCode).To(gomega.Equal(0))
}

func TestSandboxExecWithPty(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	p, err := sb.Exec(ctx, []string{"echo", "hello"}, &modal.SandboxExecParams{PTY: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	exitCode, err := p.Wait(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(exitCode).To(gomega.Equal(0))
}

func TestSandboxExecWaitTimeout(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	p, err := sb.Exec(ctx, []string{"sleep", "999"}, &modal.SandboxExecParams{Timeout: 1 * time.Second})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	exitCode, err := p.Wait(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(exitCode).To(gomega.Equal(128 + 9))
}

func TestSandboxExecOutputTimeout(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sb)

	p, err := sb.Exec(ctx, []string{"sh", "-c", "echo hi; sleep 999"}, &modal.SandboxExecParams{Timeout: 1 * time.Second})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	output, readErr := io.ReadAll(p.Stdout)
	if readErr != nil {
		g.Expect(readErr.Error()).To(gomega.ContainSubstring("deadline exceeded"))
		return
	}

	g.Expect(string(output)).To(gomega.Equal("hi\n"))

	exitCode, waitErr := p.Wait(ctx, nil)
	if waitErr != nil {
		// Deadline may have passed between stdout read completing and Wait() being called.
		g.Expect(waitErr.Error()).To(gomega.ContainSubstring("deadline exceeded"))
	} else {
		g.Expect(exitCode).To(gomega.Equal(137))
	}
}

func TestSandboxDoubleTerminateIsNotAllowed(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ToNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())

	_, err = sb.Terminate(ctx, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())

	_, err = sb.Terminate(ctx, nil)
	g.Expect(err.Error()).To(gomega.ContainSubstring("ClientClosedError: Unable to perform operation on a detached sandbox"))
}

func TestSandboxExecAfterTerminateReturnsClientClosedError(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ToNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())

	_, err = sb.Terminate(ctx, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())

	_, err = sb.Exec(ctx, []string{"echo", "hello"}, nil)
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("ClientClosedError: Unable to perform operation on a detached sandbox"))
}

func TestContainerProcessReadStdoutAfterSandboxTerminateReturnsClientClosedError(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ToNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())

	p, err := sb.Exec(ctx, []string{"sh", "-c", "echo exec-stdout; echo exec-stderr >&2"}, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())

	exitCode, err := p.Wait(ctx, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(exitCode).To(gomega.Equal(0))

	_, err = sb.Terminate(ctx, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())

	_, err = io.ReadAll(p.Stdout)
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("ClientClosedError: Unable to perform operation on a detached sandbox"))

	_, err = io.ReadAll(p.Stderr)
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("ClientClosedError: Unable to perform operation on a detached sandbox"))
}

func TestContainerProcessWriteStinAfterDetach(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{Command: []string{"cat"}})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	sbFromID, err := tc.Sandboxes.FromID(ctx, sb.SandboxID, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer terminateSandbox(g, sbFromID)

	p, err := sb.Exec(ctx, []string{"cat"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	err = sb.Detach()
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	_, err = p.Stdin.Write([]byte("this is input that should be mirrored by cat"))
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("ClientClosedError: Unable to perform operation on a detached sandbox"))
}

func TestContainerProcessReadStdoutAfterSandboxDetach(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ToNot(gomega.HaveOccurred())

	image := tc.Images.FromRegistry("alpine:3.21", nil)

	sb, err := tc.Sandboxes.Create(ctx, app, image, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())

	p, err := sb.Exec(ctx, []string{"sh", "-c", "echo exec-stdout; echo exec-stderr >&2"}, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())

	exitCode, err := p.Wait(ctx, nil)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(exitCode).To(gomega.Equal(0))

	err = sb.Detach()
	g.Expect(err).ToNot(gomega.HaveOccurred())

	_, err = io.ReadAll(p.Stdout)
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("ClientClosedError: Unable to perform operation on a detached sandbox"))

	_, err = io.ReadAll(p.Stderr)
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("ClientClosedError: Unable to perform operation on a detached sandbox"))
}
