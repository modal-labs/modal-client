package modal

import (
	"bytes"
	"testing"
	"time"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"github.com/onsi/gomega"
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

func TestTaskExecStartProto_WithoutPTY(t *testing.T) {
	g := gomega.NewWithT(t)
	req, err := buildTaskExecStartRequestProto("task-123", "exec-456", []string{"bash"}, SandboxExecParams{})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	ptyInfo := req.GetPtyInfo()
	g.Expect(ptyInfo).Should(gomega.BeNil())
}

func TestTaskExecStartProto_WithPTY(t *testing.T) {
	g := gomega.NewWithT(t)
	req, err := buildTaskExecStartRequestProto("task-123", "exec-456", []string{"bash"}, SandboxExecParams{
		PTY: true,
	})
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

	req, err := buildTaskExecStartRequestProto("task-123", "exec-456", []string{"bash"}, SandboxExecParams{})
	g.Expect(err).ToNot(gomega.HaveOccurred())

	g.Expect(req.GetWorkdir()).To(gomega.BeEmpty())
	g.Expect(req.HasTimeoutSecs()).To(gomega.BeFalse())
	g.Expect(req.GetSecretIds()).To(gomega.BeEmpty())
	g.Expect(req.GetPtyInfo()).To(gomega.BeNil())
	g.Expect(req.GetStdoutConfig()).To(gomega.Equal(pb.TaskExecStdoutConfig_TASK_EXEC_STDOUT_CONFIG_PIPE))
	g.Expect(req.GetStderrConfig()).To(gomega.Equal(pb.TaskExecStderrConfig_TASK_EXEC_STDERR_CONFIG_PIPE))
}

func TestTaskExecStartRequestProto_WithStdoutIgnore(t *testing.T) {
	g := gomega.NewWithT(t)

	req, err := buildTaskExecStartRequestProto("task-123", "exec-456", []string{"bash"}, SandboxExecParams{
		Stdout: Ignore,
	})
	g.Expect(err).ToNot(gomega.HaveOccurred())

	g.Expect(req.GetStdoutConfig()).To(gomega.Equal(pb.TaskExecStdoutConfig_TASK_EXEC_STDOUT_CONFIG_DEVNULL))
	g.Expect(req.GetStderrConfig()).To(gomega.Equal(pb.TaskExecStderrConfig_TASK_EXEC_STDERR_CONFIG_PIPE))
}

func TestTaskExecStartRequestProto_WithStderrIgnore(t *testing.T) {
	g := gomega.NewWithT(t)

	req, err := buildTaskExecStartRequestProto("task-123", "exec-456", []string{"bash"}, SandboxExecParams{
		Stderr: Ignore,
	})
	g.Expect(err).ToNot(gomega.HaveOccurred())

	g.Expect(req.GetStdoutConfig()).To(gomega.Equal(pb.TaskExecStdoutConfig_TASK_EXEC_STDOUT_CONFIG_PIPE))
	g.Expect(req.GetStderrConfig()).To(gomega.Equal(pb.TaskExecStderrConfig_TASK_EXEC_STDERR_CONFIG_DEVNULL))
}

func TestTaskExecStartRequestProto_WithWorkdir(t *testing.T) {
	g := gomega.NewWithT(t)

	req, err := buildTaskExecStartRequestProto("task-123", "exec-456", []string{"pwd"}, SandboxExecParams{
		Workdir: "/tmp",
	})
	g.Expect(err).ToNot(gomega.HaveOccurred())

	g.Expect(req.GetWorkdir()).To(gomega.Equal("/tmp"))
}

func TestTaskExecStartRequestProto_WithTimeout(t *testing.T) {
	g := gomega.NewWithT(t)
	timeout := 30 * time.Second

	req, err := buildTaskExecStartRequestProto("task-123", "exec-456", []string{"sleep", "10"}, SandboxExecParams{
		Timeout: timeout,
	})
	g.Expect(err).ToNot(gomega.HaveOccurred())

	g.Expect(req.HasTimeoutSecs()).To(gomega.BeTrue())
	g.Expect(req.GetTimeoutSecs()).To(gomega.Equal(uint32(30)))
}

func TestTaskExecStartRequestProto_InvalidTimeoutNegative(t *testing.T) {
	g := gomega.NewWithT(t)

	_, err := buildTaskExecStartRequestProto("task-123", "exec-456", []string{"echo", "hi"}, SandboxExecParams{
		Timeout: -1 * time.Second,
	})
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("must be non-negative"))
}

func TestTaskExecStartRequestProto_InvalidTimeoutNotWholeSeconds(t *testing.T) {
	g := gomega.NewWithT(t)

	_, err := buildTaskExecStartRequestProto("task-123", "exec-456", []string{"echo", "hi"}, SandboxExecParams{
		Timeout: 1500 * time.Millisecond,
	})
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
