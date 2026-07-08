package test

import (
	"errors"
	"io"
	"testing"

	modal "github.com/modal-labs/modal-client/go"
	"github.com/onsi/gomega"
)

func buildAlpineImage(t *testing.T, g *gomega.WithT, tc *modal.Client) *modal.Image {
	t.Helper()
	ctx := t.Context()

	app, err := tc.Apps.FromName(ctx, "libmodal-test", &modal.AppFromNameParams{CreateIfMissing: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	image, err := tc.Images.FromRegistry("alpine:3.21", nil).Build(ctx, app, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(image.ImageID).ShouldNot(gomega.BeEmpty())
	return image
}

func TestSidecarBasicLifecycle(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	sb := createSandbox(ctx, g, tc)
	defer terminateSandbox(g, sb)

	image := buildAlpineImage(t, g, tc)

	container, err := sb.ExperimentalSidecars.Create(ctx, "worker", image, &modal.SidecarCreateParams{
		Command: []string{"sleep", "100"},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(container.ContainerID).ShouldNot(gomega.BeEmpty())
	g.Expect(container.ContainerName).Should(gomega.Equal("worker"))

	poll, err := container.Poll(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(poll).Should(gomega.BeNil())

	exitCode, err := container.Terminate(ctx, &modal.SidecarTerminateParams{Wait: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(exitCode).Should(gomega.Equal(137))

	waitCode, err := container.Wait(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(waitCode).Should(gomega.Equal(137))

	pollAfter, err := container.Poll(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(pollAfter).ShouldNot(gomega.BeNil())
	g.Expect(*pollAfter).Should(gomega.Equal(137))

	terminated, err := sb.ExperimentalSidecars.List(ctx, &modal.SidecarListParams{IncludeTerminated: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	names := make([]string, 0, len(terminated))
	for _, c := range terminated {
		names = append(names, c.ContainerName)
	}
	g.Expect(names).Should(gomega.Equal([]string{"worker"}))
}

func TestSidecarWaitAfterNaturalExit(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	sb := createSandbox(ctx, g, tc)
	defer terminateSandbox(g, sb)

	image := buildAlpineImage(t, g, tc)

	container, err := sb.ExperimentalSidecars.Create(ctx, "oneshot", image, &modal.SidecarCreateParams{
		Command: []string{"sh", "-c", "exit 42"},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	first, err := container.Wait(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(first).Should(gomega.Equal(42))

	second, err := container.Wait(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(second).Should(gomega.Equal(42))

	_, err = sb.ExperimentalSidecars.Get(ctx, "oneshot", nil)
	g.Expect(err).Should(gomega.HaveOccurred())
	var notFound modal.NotFoundError
	g.Expect(errors.As(err, &notFound)).Should(gomega.BeTrue())

	got, err := sb.ExperimentalSidecars.Get(ctx, "oneshot", &modal.SidecarGetParams{IncludeTerminated: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(got.ContainerID).Should(gomega.Equal(container.ContainerID))

	replacement, err := sb.ExperimentalSidecars.Create(ctx, "oneshot", image, &modal.SidecarCreateParams{
		Command: []string{"sleep", "100"},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(replacement.ContainerID).ShouldNot(gomega.Equal(container.ContainerID))

	listed, err := sb.ExperimentalSidecars.List(ctx, &modal.SidecarListParams{IncludeTerminated: true})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	ids := make([]string, 0, len(listed))
	for _, c := range listed {
		ids = append(ids, c.ContainerID)
	}
	g.Expect(ids).Should(gomega.ContainElements(container.ContainerID, replacement.ContainerID))
}

func TestSidecarCreateRejectsMainName(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	sb := createSandbox(ctx, g, tc)
	defer terminateSandbox(g, sb)

	image := buildAlpineImage(t, g, tc)

	_, err := sb.ExperimentalSidecars.Create(ctx, "main", image, &modal.SidecarCreateParams{Command: []string{"sleep", "100"}})
	g.Expect(err).Should(gomega.HaveOccurred())
	var invalid modal.InvalidError
	g.Expect(errors.As(err, &invalid)).Should(gomega.BeTrue(), "expected InvalidError, got %T: %v", err, err)

	_, err = sb.ExperimentalSidecars.Create(ctx, "", image, &modal.SidecarCreateParams{Command: []string{"sleep", "100"}})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(errors.As(err, &invalid)).Should(gomega.BeTrue())

	_, err = sb.ExperimentalSidecars.Get(ctx, "main", nil)
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(errors.As(err, &invalid)).Should(gomega.BeTrue())
}

func TestSidecarCreateImageMustBeBuilt(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	sb := createSandbox(ctx, g, tc)
	defer terminateSandbox(g, sb)

	unbuilt := tc.Images.FromRegistry("alpine:3.21", nil)
	g.Expect(unbuilt.ImageID).Should(gomega.BeEmpty())

	_, err := sb.ExperimentalSidecars.Create(ctx, "worker", unbuilt, &modal.SidecarCreateParams{Command: []string{"sleep", "100"}})
	g.Expect(err).Should(gomega.HaveOccurred())
	var invalid modal.InvalidError
	g.Expect(errors.As(err, &invalid)).Should(gomega.BeTrue())
}

func TestSidecarCreateRejectsInvalidEnvVarName(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	sb := createSandbox(ctx, g, tc)
	defer terminateSandbox(g, sb)

	image := buildAlpineImage(t, g, tc)

	_, err := sb.ExperimentalSidecars.Create(ctx, "worker", image, &modal.SidecarCreateParams{
		Command: []string{"sleep", "100"},
		Env:     map[string]string{"1INVALID": "value"},
	})
	g.Expect(err).Should(gomega.HaveOccurred())
	var invalid modal.InvalidError
	g.Expect(errors.As(err, &invalid)).Should(gomega.BeTrue(), "expected InvalidError, got %T: %v", err, err)
	g.Expect(err).Should(gomega.MatchError(gomega.ContainSubstring("is invalid for environment variables")))
}

func TestSidecarCreateForwardsSecretsAndEnv(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	secret, err := tc.Secrets.FromMap(ctx, map[string]string{"API_KEY": "secret-value"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	sb := createSandbox(ctx, g, tc)
	defer terminateSandbox(g, sb)

	image := buildAlpineImage(t, g, tc)

	container, err := sb.ExperimentalSidecars.Create(ctx, "worker", image, &modal.SidecarCreateParams{
		Command: []string{"sleep", "100"},
		Env:     map[string]string{"API_KEY": "override", "PLAIN_ENV": "plain"},
		Secrets: []*modal.Secret{secret},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	proc, err := container.Exec(ctx, []string{"sh", "-c", "printf '%s:%s' \"$API_KEY\" \"$PLAIN_ENV\""}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	output, err := io.ReadAll(proc.Stdout)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	code, err := proc.Wait(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(code).Should(gomega.Equal(0))
	g.Expect(string(output)).Should(gomega.Equal("override:plain"))
}

func TestSidecarExec(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	sb := createSandbox(ctx, g, tc)
	defer terminateSandbox(g, sb)

	image := buildAlpineImage(t, g, tc)

	container, err := sb.ExperimentalSidecars.Create(ctx, "worker", image, &modal.SidecarCreateParams{
		Command: []string{"sleep", "100"},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	proc, err := container.Exec(ctx, []string{"echo", "hello"}, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	output, err := io.ReadAll(proc.Stdout)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(string(output)).Should(gomega.Equal("hello\n"))

	code, err := proc.Wait(ctx, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(code).Should(gomega.Equal(0))
}

func TestSidecarFilesystem(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()
	tc := newTestClient(t)

	sb := createSandbox(ctx, g, tc)
	defer terminateSandbox(g, sb)

	image := buildAlpineImage(t, g, tc)

	container, err := sb.ExperimentalSidecars.Create(ctx, "worker", image, &modal.SidecarCreateParams{
		Command: []string{"sleep", "100"},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	err = container.Filesystem.WriteText(ctx, "hi from sidecar", "/tmp/sidecar-hello", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	text, err := container.Filesystem.ReadText(ctx, "/tmp/sidecar-hello", nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(text).Should(gomega.Equal("hi from sidecar"))

	// The main container should not see the file in the sidecar's filesystem.
	_, err = sb.Filesystem.Stat(ctx, "/tmp/sidecar-hello", nil)
	g.Expect(err).Should(gomega.HaveOccurred())
	var notFound modal.SandboxFilesystemNotFoundError
	g.Expect(errors.As(err, &notFound)).Should(gomega.BeTrue())
}
