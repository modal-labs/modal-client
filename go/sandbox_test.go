package modal

import (
	"bytes"
	"errors"
	"testing"
	"time"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"github.com/onsi/gomega"
)

const (
	testV1SandboxID = "sb-nGEijt9WbBMlGrsPH9FOaC"
	testV2SandboxID = "sb-01ARZ3NDEKTSV4RRFFQ69G5FAV"
)

func TestSandboxCreateRequestProto_WithoutPTY(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	req, err := buildSandboxCreateRequestProto("app-123", "img-456", SandboxCreateParams{})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	definition := req.GetDefinition()
	ptyInfo := definition.GetPtyInfo()
	g.Expect(ptyInfo).Should(gomega.BeNil())
}

func TestSandboxCreateV2RequestProto(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	req, err := buildSandboxCreateV2RequestProto("app-123", "img-456", SandboxCreateParams{
		Command: []string{"sleep", "60"},
		Timeout: 10 * time.Minute,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(req.GetAppId()).To(gomega.Equal("app-123"))
	g.Expect(req.GetDefinition().GetImageId()).To(gomega.Equal("img-456"))
	g.Expect(req.GetDefinition().GetEntrypointArgs()).To(gomega.Equal([]string{"sleep", "60"}))
	g.Expect(req.GetDefinition().GetTimeoutSecs()).To(gomega.Equal(uint32(600)))
}

func TestSandboxCreateV2RequestProto_UnsupportedOptions(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		params  SandboxCreateParams
		wantErr string
	}{
		{
			name:    "tags",
			params:  SandboxCreateParams{Tags: map[string]string{"key": "value"}},
			wantErr: "tags are not supported",
		},
		{
			name:    "gpu",
			params:  SandboxCreateParams{GPU: "A10G"},
			wantErr: "GPUs are not supported",
		},
		{
			name:    "custom domain",
			params:  SandboxCreateParams{CustomDomain: "example.com"},
			wantErr: "custom domains are not supported",
		},
		{
			name:    "proxy",
			params:  SandboxCreateParams{Proxy: &Proxy{ProxyID: "pr-123"}},
			wantErr: "proxies are not supported",
		},
		{
			name:    "readiness probe",
			params:  SandboxCreateParams{ReadinessProbe: &Probe{}},
			wantErr: "readiness probes are not supported",
		},
		{
			name:    "include oidc identity token",
			params:  SandboxCreateParams{IncludeOidcIdentityToken: true},
			wantErr: "IncludeOidcIdentityToken is not supported",
		},
		{
			name: "cloud bucket mount with oidc auth role",
			params: func() SandboxCreateParams {
				role := "arn:aws:iam::123:role/r"
				return SandboxCreateParams{
					CloudBucketMounts: map[string]*CloudBucketMount{
						"/bucket": {BucketName: "bucket", OidcAuthRoleArn: &role},
					},
				}
			}(),
			wantErr: "CloudBucketMount with OidcAuthRoleArn is not supported",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			g := gomega.NewWithT(t)

			_, err := buildSandboxCreateV2RequestProto("app-123", "img-456", tt.params)
			g.Expect(err).Should(gomega.HaveOccurred())
			g.Expect(err.Error()).To(gomega.ContainSubstring(tt.wantErr))
		})
	}
}

func TestSandboxCreateV2RequestProto_VolumesAndCloudBucketMounts(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	req, err := buildSandboxCreateV2RequestProto("app-123", "img-456", SandboxCreateParams{
		Volumes:           map[string]*Volume{"/mnt/vol": {VolumeID: "vo-123"}},
		CloudBucketMounts: map[string]*CloudBucketMount{"/mnt/s3": {BucketName: "my-bucket"}},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	volumeMounts := req.GetDefinition().GetVolumeMounts()
	g.Expect(volumeMounts).To(gomega.HaveLen(1))
	g.Expect(volumeMounts[0].GetMountPath()).To(gomega.Equal("/mnt/vol"))
	g.Expect(volumeMounts[0].GetVolumeId()).To(gomega.Equal("vo-123"))

	cloudBucketMounts := req.GetDefinition().GetCloudBucketMounts()
	g.Expect(cloudBucketMounts).To(gomega.HaveLen(1))
	g.Expect(cloudBucketMounts[0].GetMountPath()).To(gomega.Equal("/mnt/s3"))
	g.Expect(cloudBucketMounts[0].GetBucketName()).To(gomega.Equal("my-bucket"))
}

func TestGetSandboxVersion(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		sandboxID string
		want      sandboxVersion
	}{
		{name: "v1", sandboxID: testV1SandboxID, want: sandboxVersionV1},
		{name: "v2", sandboxID: testV2SandboxID, want: sandboxVersionV2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			g := gomega.NewWithT(t)

			got, err := getSandboxVersion(tt.sandboxID)
			g.Expect(err).ShouldNot(gomega.HaveOccurred())
			g.Expect(got).To(gomega.Equal(tt.want))
		})
	}
}

func TestGetSandboxVersionRejectsInvalidID(t *testing.T) {
	t.Parallel()

	tests := []string{
		"sb-123",
		"sb-nGEijt9WbBMlGrsPH9FOa_",
		"sb-81ARZ3NDEKTSV4RRFFQ69G5FAV",
		"sb-01arz3ndektsv4rrffq69g5fav",
		"fu-01ARZ3NDEKTSV4RRFFQ69G5FAV",
		"sb-foo-bar",
		"not-a-sandbox-id",
	}

	for _, sandboxID := range tests {
		t.Run(sandboxID, func(t *testing.T) {
			t.Parallel()
			g := gomega.NewWithT(t)

			_, err := getSandboxVersion(sandboxID)
			g.Expect(err).Should(gomega.HaveOccurred())
			g.Expect(err.Error()).To(gomega.ContainSubstring("Invalid Sandbox ID"))
		})
	}
}

func TestSandboxV2UnsupportedRuntimeMethods(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)
	ctx := t.Context()

	sb := newSandbox(&Client{}, "sb-v2-123")
	sb.isV2 = true
	sb.taskID = "ta-v2-123"

	wantErr := "not supported for V2 sandboxes"

	_, err := sb.CreateConnectToken(ctx, nil)
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring(wantErr))

	err = sb.MountImage(ctx, "/mnt", nil, nil)
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring(wantErr))

	err = sb.UnmountImage(ctx, "/mnt", nil)
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring(wantErr))

	_, err = sb.SnapshotDirectory(ctx, "/mnt", nil)
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring(wantErr))

	err = sb.SetTags(ctx, map[string]string{}, nil)
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring(wantErr))

	_, err = sb.GetTags(ctx, nil)
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring(wantErr))

	err = sb.WaitUntilReady(ctx, time.Second, nil)
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring(wantErr))
}

func TestSandboxV2StdioUnsupported(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	sb := newSandboxV2(&Client{}, "sb-v2-123", "ta-v2-123")
	wantErr := "not supported for V2 sandboxes"

	_, err := sb.Stdin.Write([]byte("hello"))
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("Sandbox.Stdin"))
	g.Expect(err.Error()).To(gomega.ContainSubstring(wantErr))
	var invalidErr InvalidError
	g.Expect(errors.As(err, &invalidErr)).To(gomega.BeTrue())

	err = sb.Stdin.Close()
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("Sandbox.Stdin"))
	g.Expect(err.Error()).To(gomega.ContainSubstring(wantErr))

	buf := make([]byte, 1)
	_, err = sb.Stdout.Read(buf)
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("Sandbox.Stdout"))
	g.Expect(err.Error()).To(gomega.ContainSubstring(wantErr))

	err = sb.Stdout.Close()
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("Sandbox.Stdout"))
	g.Expect(err.Error()).To(gomega.ContainSubstring(wantErr))

	_, err = sb.Stderr.Read(buf)
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("Sandbox.Stderr"))
	g.Expect(err.Error()).To(gomega.ContainSubstring(wantErr))

	err = sb.Stderr.Close()
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("Sandbox.Stderr"))
	g.Expect(err.Error()).To(gomega.ContainSubstring(wantErr))
}

func TestSandboxCreateRequestProto_WithPTY(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	req, err := buildSandboxCreateRequestProto("app-123", "img-456", SandboxCreateParams{
		PTY: true,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	definition := req.GetDefinition()
	ptyInfo := definition.GetPtyInfo()
	g.Expect(ptyInfo.GetEnabled()).To(gomega.BeTrue())
	g.Expect(ptyInfo.GetWinszRows()).To(gomega.Equal(uint32(24)))
	g.Expect(ptyInfo.GetWinszCols()).To(gomega.Equal(uint32(80)))
	g.Expect(ptyInfo.GetEnvTerm()).To(gomega.Equal("xterm-256color"))
	g.Expect(ptyInfo.GetEnvColorterm()).To(gomega.Equal("truecolor"))
	g.Expect(ptyInfo.GetPtyType()).To(gomega.Equal(pb.PTYInfo_PTY_TYPE_SHELL))
}

func TestProbeWithTCPBadValues(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	_, err := NewTCPProbe(0, nil)
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("expects port in [1, 65535]"))

	_, err = NewTCPProbe(8080, &TCPProbeParams{Interval: -1 * time.Millisecond})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("expects interval > 0"))

	_, err = NewTCPProbe(8080, &TCPProbeParams{Interval: 1500 * time.Microsecond})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("whole number of milliseconds"))
}

func TestProbeWithExecBadValues(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	_, err := NewExecProbe(nil, nil)
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("requires at least one argument"))

	_, err = NewExecProbe([]string{"echo"}, &ExecProbeParams{Interval: -1 * time.Millisecond})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("expects interval > 0"))
}

func TestSandboxCreateRequestProto_WithReadinessProbeTCP(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	probe, err := NewTCPProbe(8080, &TCPProbeParams{Interval: 250 * time.Millisecond})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	req, err := buildSandboxCreateRequestProto("app-123", "img-456", SandboxCreateParams{
		ReadinessProbe: probe,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	readinessProbe := req.GetDefinition().GetReadinessProbe()
	g.Expect(readinessProbe).ShouldNot(gomega.BeNil())
	g.Expect(readinessProbe.GetTcpPort()).To(gomega.Equal(uint32(8080)))
	g.Expect(readinessProbe.GetIntervalMs()).To(gomega.Equal(uint32(250)))
}

func TestSandboxCreateRequestProto_WithReadinessProbeExec(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	probe, err := NewExecProbe([]string{"sh", "-c", "echo ok"}, &ExecProbeParams{Interval: 300 * time.Millisecond})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	req, err := buildSandboxCreateRequestProto("app-123", "img-456", SandboxCreateParams{
		ReadinessProbe: probe,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	readinessProbe := req.GetDefinition().GetReadinessProbe()
	g.Expect(readinessProbe).ShouldNot(gomega.BeNil())
	g.Expect(readinessProbe.GetExecCommand().GetArgv()).To(gomega.Equal([]string{"sh", "-c", "echo ok"}))
	g.Expect(readinessProbe.GetIntervalMs()).To(gomega.Equal(uint32(300)))
}

func TestTaskExecStartProto_WithoutPTY(t *testing.T) {
	g := gomega.NewWithT(t)
	req, err := buildTaskExecStartRequestProto("task-123", "exec-456", []string{"bash"}, SandboxExecParams{}, "")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	ptyInfo := req.GetPtyInfo()
	g.Expect(ptyInfo).Should(gomega.BeNil())
}

func TestTaskExecStartProto_WithPTY(t *testing.T) {
	g := gomega.NewWithT(t)
	req, err := buildTaskExecStartRequestProto("task-123", "exec-456", []string{"bash"}, SandboxExecParams{
		PTY: true,
	}, "")
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	ptyInfo := req.GetPtyInfo()
	g.Expect(ptyInfo).ShouldNot(gomega.BeNil())
	g.Expect(ptyInfo.GetEnabled()).To(gomega.BeTrue())
	g.Expect(ptyInfo.GetWinszRows()).To(gomega.Equal(uint32(24)))
	g.Expect(ptyInfo.GetWinszCols()).To(gomega.Equal(uint32(80)))
	g.Expect(ptyInfo.GetEnvTerm()).To(gomega.Equal("xterm-256color"))
	g.Expect(ptyInfo.GetEnvColorterm()).To(gomega.Equal("truecolor"))
	g.Expect(ptyInfo.GetPtyType()).To(gomega.Equal(pb.PTYInfo_PTY_TYPE_SHELL))
	g.Expect(ptyInfo.GetNoTerminateOnIdleStdin()).To(gomega.BeTrue())
}

func TestTaskExecStartRequestProto_DefaultValues(t *testing.T) {
	g := gomega.NewWithT(t)

	req, err := buildTaskExecStartRequestProto("task-123", "exec-456", []string{"bash"}, SandboxExecParams{}, "")
	g.Expect(err).ToNot(gomega.HaveOccurred())

	g.Expect(req.GetWorkdir()).To(gomega.BeEmpty())
	g.Expect(req.HasTimeoutSecs()).To(gomega.BeFalse())
	g.Expect(req.GetSecretIds()).To(gomega.BeEmpty())
	g.Expect(req.GetEnv()).To(gomega.BeEmpty())
	g.Expect(req.GetPtyInfo()).To(gomega.BeNil())
	g.Expect(req.GetStdoutConfig()).To(gomega.Equal(pb.TaskExecStdoutConfig_TASK_EXEC_STDOUT_CONFIG_PIPE))
	g.Expect(req.GetStderrConfig()).To(gomega.Equal(pb.TaskExecStderrConfig_TASK_EXEC_STDERR_CONFIG_PIPE))
}

func TestTaskExecStartRequestProto_WithStdoutIgnore(t *testing.T) {
	g := gomega.NewWithT(t)

	req, err := buildTaskExecStartRequestProto("task-123", "exec-456", []string{"bash"}, SandboxExecParams{
		Stdout: Ignore,
	}, "")
	g.Expect(err).ToNot(gomega.HaveOccurred())

	g.Expect(req.GetStdoutConfig()).To(gomega.Equal(pb.TaskExecStdoutConfig_TASK_EXEC_STDOUT_CONFIG_DEVNULL))
	g.Expect(req.GetStderrConfig()).To(gomega.Equal(pb.TaskExecStderrConfig_TASK_EXEC_STDERR_CONFIG_PIPE))
}

func TestTaskExecStartRequestProto_WithStderrIgnore(t *testing.T) {
	g := gomega.NewWithT(t)

	req, err := buildTaskExecStartRequestProto("task-123", "exec-456", []string{"bash"}, SandboxExecParams{
		Stderr: Ignore,
	}, "")
	g.Expect(err).ToNot(gomega.HaveOccurred())

	g.Expect(req.GetStdoutConfig()).To(gomega.Equal(pb.TaskExecStdoutConfig_TASK_EXEC_STDOUT_CONFIG_PIPE))
	g.Expect(req.GetStderrConfig()).To(gomega.Equal(pb.TaskExecStderrConfig_TASK_EXEC_STDERR_CONFIG_DEVNULL))
}

func TestTaskExecStartRequestProto_WithWorkdir(t *testing.T) {
	tests := []struct {
		name        string
		workdir     string
		expected    string
		expectedErr string
	}{
		{
			name:     "absolute",
			workdir:  "/tmp",
			expected: "/tmp",
		},
		{
			name:        "relative",
			workdir:     "tmp",
			expectedErr: "workdir must be an absolute path",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewWithT(t)

			req, err := buildTaskExecStartRequestProto("task-123", "exec-456", []string{"pwd"}, SandboxExecParams{
				Workdir: tt.workdir,
			}, "")
			if tt.expectedErr != "" {
				g.Expect(err).To(gomega.HaveOccurred())
				g.Expect(err.Error()).To(gomega.ContainSubstring(tt.expectedErr))
				return
			}

			g.Expect(err).ToNot(gomega.HaveOccurred())
			g.Expect(req.GetWorkdir()).To(gomega.Equal(tt.expected))
		})
	}
}

func TestTaskExecStartRequestProto_WithTimeout(t *testing.T) {
	g := gomega.NewWithT(t)
	timeout := 30 * time.Second

	req, err := buildTaskExecStartRequestProto("task-123", "exec-456", []string{"sleep", "10"}, SandboxExecParams{
		Timeout: timeout,
	}, "")
	g.Expect(err).ToNot(gomega.HaveOccurred())

	g.Expect(req.HasTimeoutSecs()).To(gomega.BeTrue())
	g.Expect(req.GetTimeoutSecs()).To(gomega.Equal(uint32(30)))
}

func TestTaskExecStartRequestProto_WithEnv(t *testing.T) {
	g := gomega.NewWithT(t)

	req, err := buildTaskExecStartRequestProto("task-123", "exec-456", []string{"env"}, SandboxExecParams{
		Env: map[string]string{"FOO": "bar"},
	}, "")
	g.Expect(err).ToNot(gomega.HaveOccurred())

	g.Expect(req.GetEnv()).To(gomega.Equal(map[string]string{"FOO": "bar"}))
}

func TestTaskExecStartRequestProto_InvalidTimeoutNegative(t *testing.T) {
	g := gomega.NewWithT(t)

	_, err := buildTaskExecStartRequestProto("task-123", "exec-456", []string{"echo", "hi"}, SandboxExecParams{
		Timeout: -1 * time.Second,
	}, "")
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("must be non-negative"))
}

func TestTaskExecStartRequestProto_InvalidTimeoutNotWholeSeconds(t *testing.T) {
	g := gomega.NewWithT(t)

	_, err := buildTaskExecStartRequestProto("task-123", "exec-456", []string{"echo", "hi"}, SandboxExecParams{
		Timeout: 1500 * time.Millisecond,
	}, "")
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("whole number of seconds"))
}

func TestValidateExecArgsWithArgsWithinLimit(t *testing.T) {
	g := gomega.NewWithT(t)

	err := ValidateExecArgs([]string{"echo", "hello"})
	g.Expect(err).ToNot(gomega.HaveOccurred())
}

func TestValidateExecArgsWithArgsExceedingArgMax(t *testing.T) {
	g := gomega.NewWithT(t)

	largeArg := bytes.Repeat([]byte{'a'}, 1<<16+1)

	err := ValidateExecArgs([]string{string(largeArg)})
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("Total length of CMD arguments must be less than"))
}

func TestSandboxCreateRequestProto_WithCPUAndCPULimit(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	req, err := buildSandboxCreateRequestProto("app-123", "img-456", SandboxCreateParams{
		CPU:      2.0,
		CPULimit: 4.5,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	resources := req.GetDefinition().GetResources()
	g.Expect(resources.GetMilliCpu()).To(gomega.Equal(uint32(2000)))
	g.Expect(resources.GetMilliCpuMax()).To(gomega.Equal(uint32(4500)))
}

func TestSandboxCreateRequestProto_CPULimitLowerThanCPU(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	_, err := buildSandboxCreateRequestProto("app-123", "img-456", SandboxCreateParams{
		CPU:      4.0,
		CPULimit: 2.0,
	})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("the CPU request (4.000000) cannot be higher than CPULimit (2.000000)"))
}

func TestSandboxCreateRequestProto_CPULimitWithoutCPU(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	_, err := buildSandboxCreateRequestProto("app-123", "img-456", SandboxCreateParams{
		CPULimit: 4.0,
	})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("must also specify non-zero CPU request when CPULimit is specified"))
}

func TestSandboxCreateRequestProto_WithMemoryAndMemoryLimit(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	req, err := buildSandboxCreateRequestProto("app-123", "img-456", SandboxCreateParams{
		MemoryMiB:      1024,
		MemoryLimitMiB: 2048,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	resources := req.GetDefinition().GetResources()
	g.Expect(resources.GetMemoryMb()).To(gomega.Equal(uint32(1024)))
	g.Expect(resources.GetMemoryMbMax()).To(gomega.Equal(uint32(2048)))
}

func TestSandboxCreateRequestProto_MemoryLimitLowerThanMemory(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	_, err := buildSandboxCreateRequestProto("app-123", "img-456", SandboxCreateParams{
		MemoryMiB:      2048,
		MemoryLimitMiB: 1024,
	})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("the MemoryMiB request (2048) cannot be higher than MemoryLimitMiB (1024)"))
}

func TestSandboxCreateRequestProto_MemoryLimitWithoutMemory(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	_, err := buildSandboxCreateRequestProto("app-123", "img-456", SandboxCreateParams{
		MemoryLimitMiB: 2048,
	})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("must also specify non-zero MemoryMiB request when MemoryLimitMiB is specified"))
}

func TestSandboxCreateRequestProto_NegativeCPU(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	_, err := buildSandboxCreateRequestProto("app-123", "img-456", SandboxCreateParams{
		CPU: -1.0,
	})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("must be a positive number"))
}

func TestSandboxCreateRequestProto_NegativeMemory(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	_, err := buildSandboxCreateRequestProto("app-123", "img-456", SandboxCreateParams{
		MemoryMiB: -100,
	})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("must be a positive number"))
}

func TestSandboxCreateRequestProto_DefaultValues(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	req, err := buildSandboxCreateRequestProto("app-123", "img-456", SandboxCreateParams{})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	def := req.GetDefinition()
	g.Expect(def.GetTimeoutSecs()).To(gomega.Equal(uint32(300)))
	g.Expect(def.GetEntrypointArgs()).To(gomega.BeEmpty())
	g.Expect(def.GetNetworkAccess().GetNetworkAccessType()).To(gomega.Equal(pb.NetworkAccess_OPEN))
	g.Expect(def.GetNetworkAccess().GetAllowedCidrs()).To(gomega.BeEmpty())
	g.Expect(def.GetVerbose()).To(gomega.BeFalse())
	g.Expect(def.GetCloudProviderStr()).To(gomega.BeEmpty())
	g.Expect(def.GetResources().GetMilliCpu()).To(gomega.Equal(uint32(0)))
	g.Expect(def.GetResources().GetMemoryMb()).To(gomega.Equal(uint32(0)))
	g.Expect(def.GetPtyInfo()).To(gomega.BeNil())
	g.Expect(def.HasIdleTimeoutSecs()).To(gomega.BeFalse())
	g.Expect(def.GetWorkdir()).To(gomega.BeEmpty())
	g.Expect(def.GetSchedulerPlacement()).To(gomega.BeNil())
	g.Expect(def.GetProxyId()).To(gomega.BeEmpty())
	g.Expect(def.GetVolumeMounts()).To(gomega.BeEmpty())
	g.Expect(def.GetCloudBucketMounts()).To(gomega.BeEmpty())
	g.Expect(def.GetSecretIds()).To(gomega.BeEmpty())
	g.Expect(def.GetOpenPorts().GetPorts()).To(gomega.BeEmpty())
	g.Expect(def.GetName()).To(gomega.Equal(""))
	g.Expect(def.GetIncludeOidcIdentityToken()).To(gomega.BeFalse())
	g.Expect(req.GetTags()).To(gomega.BeEmpty())
}

func TestSandboxCreateRequestProto_CustomDomain(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	req, err := buildSandboxCreateRequestProto("app-123", "img-456", SandboxCreateParams{
		CustomDomain: "example.com",
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	def := req.GetDefinition()
	g.Expect(def.GetCustomDomain()).To(gomega.Equal("example.com"))
}

func TestSandboxCreateRequestProto_IncludeOidcIdentityToken(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	req, err := buildSandboxCreateRequestProto("app-123", "img-456", SandboxCreateParams{
		IncludeOidcIdentityToken: true,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	def := req.GetDefinition()
	g.Expect(def.GetIncludeOidcIdentityToken()).To(gomega.BeTrue())
}

func TestSandboxCreateRequestProto_OutboundCIDRAllowlist(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	// OutboundCIDRAllowlist is reflected in the network access proto.
	req, err := buildSandboxCreateRequestProto("app-123", "img-456", SandboxCreateParams{
		OutboundCIDRAllowlist: []string{"10.0.0.0/8", "192.168.0.0/16"},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	def := req.GetDefinition()
	g.Expect(def.GetNetworkAccess().GetNetworkAccessType()).To(gomega.Equal(pb.NetworkAccess_ALLOWLIST))
	g.Expect(def.GetNetworkAccess().GetAllowedCidrs()).To(gomega.Equal([]string{"10.0.0.0/8", "192.168.0.0/16"}))

	// Cannot be combined with BlockNetwork.
	_, err = buildSandboxCreateRequestProto("app-123", "img-456", SandboxCreateParams{
		BlockNetwork:          true,
		OutboundCIDRAllowlist: []string{"10.0.0.0/8"},
	})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("OutboundCIDRAllowlist cannot be used when BlockNetwork is enabled"))
}

func TestSandboxCreateRequestProto_OutboundDomainAllowlist(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	// Domain-only allowlist.
	req, err := buildSandboxCreateRequestProto("app-123", "img-456", SandboxCreateParams{
		OutboundDomainAllowlist: []string{"example.com", "*.github.com"},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	def := req.GetDefinition()
	g.Expect(def.GetNetworkAccess().GetNetworkAccessType()).To(gomega.Equal(pb.NetworkAccess_ALLOWLIST))
	g.Expect(def.GetNetworkAccess().GetAllowedDomains()).To(gomega.Equal([]string{"example.com", "*.github.com"}))
	g.Expect(def.GetNetworkAccess().GetAllowedCidrs()).To(gomega.BeNil())

	// Domain + CIDR combined.
	req, err = buildSandboxCreateRequestProto("app-123", "img-456", SandboxCreateParams{
		OutboundDomainAllowlist: []string{"api.example.com"},
		OutboundCIDRAllowlist:   []string{"10.0.0.0/8"},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	def = req.GetDefinition()
	g.Expect(def.GetNetworkAccess().GetNetworkAccessType()).To(gomega.Equal(pb.NetworkAccess_ALLOWLIST))
	g.Expect(def.GetNetworkAccess().GetAllowedDomains()).To(gomega.Equal([]string{"api.example.com"}))
	g.Expect(def.GetNetworkAccess().GetAllowedCidrs()).To(gomega.Equal([]string{"10.0.0.0/8"}))

	// Cannot be combined with BlockNetwork.
	_, err = buildSandboxCreateRequestProto("app-123", "img-456", SandboxCreateParams{
		BlockNetwork:            true,
		OutboundDomainAllowlist: []string{"example.com"},
	})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("OutboundDomainAllowlist cannot be used when BlockNetwork is enabled"))
}

func TestSandboxCreateRequestProto_InboundCIDRAllowlist(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	// InboundCIDRAllowlist is set on the definition.
	req, err := buildSandboxCreateRequestProto("app-123", "img-456", SandboxCreateParams{
		InboundCIDRAllowlist: []string{"10.0.0.0/8", "192.168.0.0/16"},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(req.GetDefinition().GetInboundCidrAllowlist()).To(gomega.Equal([]string{"10.0.0.0/8", "192.168.0.0/16"}))

	// Empty by default.
	req, err = buildSandboxCreateRequestProto("app-123", "img-456", SandboxCreateParams{})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(req.GetDefinition().GetInboundCidrAllowlist()).To(gomega.BeEmpty())

	// Cannot be combined with BlockNetwork.
	_, err = buildSandboxCreateRequestProto("app-123", "img-456", SandboxCreateParams{
		BlockNetwork:         true,
		InboundCIDRAllowlist: []string{"10.0.0.0/8"},
	})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("InboundCIDRAllowlist cannot be used when BlockNetwork is enabled"))
}

func TestSandboxCreateRequestProto_WithTags(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	req, err := buildSandboxCreateRequestProto("app-123", "img-456", SandboxCreateParams{
		Tags: map[string]string{"env": "prod", "team": "infra"},
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	got := map[string]string{}
	for _, tag := range req.GetTags() {
		got[tag.GetTagName()] = tag.GetTagValue()
	}
	g.Expect(got).To(gomega.Equal(map[string]string{"env": "prod", "team": "infra"}))
}
