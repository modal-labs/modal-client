package modal

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"iter"
	"log/slog"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/djherbis/buffer"
	"github.com/djherbis/nio/v3"
	"github.com/google/uuid"
	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// SandboxService provides Sandbox related operations.
type SandboxService interface {
	Create(ctx context.Context, app *App, image *Image, params *SandboxCreateParams) (*Sandbox, error)
	FromID(ctx context.Context, sandboxID string) (*Sandbox, error)
	FromName(ctx context.Context, appName, name string, params *SandboxFromNameParams) (*Sandbox, error)
	List(ctx context.Context, params *SandboxListParams) (iter.Seq2[*Sandbox, error], error)
}

type sandboxServiceImpl struct{ client *Client }

const (
	defaultProbeInterval                = 100 * time.Millisecond
	maxProbeIntervalMilliseconds uint64 = ^uint64(0) >> 32
)

// Probe configures a sandbox readiness probe.
type Probe struct {
	tcpPort    *uint32
	execArgv   []string
	intervalMs uint32
}

type TCPProbeParams struct {
	Interval time.Duration
}

// NewTCPProbe creates a TCP readiness probe.
func NewTCPProbe(port int, params *TCPProbeParams) (*Probe, error) {
	if params == nil {
		params = &TCPProbeParams{}
	}
	if params.Interval == 0 {
		params.Interval = defaultProbeInterval
	}
	if port <= 0 || port > 65535 {
		return nil, InvalidError{Exception: fmt.Sprintf("NewTCPProbe expects port in [1, 65535], got %d", port)}
	}
	intervalMs, err := validateProbeIntervalMs(params.Interval, "NewTCPProbe")
	if err != nil {
		return nil, err
	}
	tcpPort := uint32(port)
	return &Probe{
		tcpPort:    &tcpPort,
		intervalMs: intervalMs,
	}, nil
}

type ExecProbeParams struct {
	Interval time.Duration
}

// NewExecProbe creates an exec readiness probe.
func NewExecProbe(argv []string, params *ExecProbeParams) (*Probe, error) {
	if params == nil {
		params = &ExecProbeParams{}
	}
	if params.Interval == 0 {
		params.Interval = defaultProbeInterval
	}
	if len(argv) == 0 {
		return nil, InvalidError{Exception: "NewExecProbe requires at least one argument"}
	}
	intervalMs, err := validateProbeIntervalMs(params.Interval, "NewExecProbe")
	if err != nil {
		return nil, err
	}
	return &Probe{
		execArgv:   append([]string(nil), argv...),
		intervalMs: intervalMs,
	}, nil
}

func validateProbeIntervalMs(interval time.Duration, name string) (uint32, error) {
	if interval <= 0 {
		return 0, InvalidError{Exception: fmt.Sprintf("%s expects interval > 0, got %v", name, interval)}
	}
	if interval%time.Millisecond != 0 {
		return 0, InvalidError{Exception: fmt.Sprintf("%s expects interval to be a whole number of milliseconds, got %v", name, interval)}
	}
	intervalMs := uint64(interval / time.Millisecond)
	if intervalMs > maxProbeIntervalMilliseconds {
		return 0, InvalidError{Exception: fmt.Sprintf("%s interval is too large, got %v", name, interval)}
	}
	return uint32(intervalMs), nil
}

func (p *Probe) toProto() (*pb.Probe, error) {
	if p == nil {
		return nil, nil
	}
	if (p.tcpPort == nil) == (p.execArgv == nil) {
		return nil, InvalidError{Exception: "Probe must be created with NewTCPProbe(...) or NewExecProbe(...)"}
	}
	if p.intervalMs == 0 {
		return nil, InvalidError{Exception: "Probe interval must be greater than 0"}
	}

	intervalMs := p.intervalMs
	if p.tcpPort != nil {
		return pb.Probe_builder{
			TcpPort:    p.tcpPort,
			IntervalMs: &intervalMs,
		}.Build(), nil
	}

	return pb.Probe_builder{
		ExecCommand: pb.Probe_ExecCommand_builder{Argv: p.execArgv}.Build(),
		IntervalMs:  &intervalMs,
	}.Build(), nil
}

// SandboxCreateParams are options for creating a Modal Sandbox.
type SandboxCreateParams struct {
	CPU                      float64                      // CPU request in fractional, physical cores.
	CPULimit                 float64                      // Hard limit in fractional, physical CPU cores. Zero means no limit.
	MemoryMiB                int                          // Memory request in MiB.
	MemoryLimitMiB           int                          // Hard memory limit in MiB. Zero means no limit.
	GPU                      string                       // GPU reservation for the Sandbox (e.g. "A100", "T4:2", "A100-80GB:4").
	Timeout                  time.Duration                // Maximum lifetime of the Sandbox. Defaults to 5 minutes. If you pass zero you get the default 5 minutes.
	IdleTimeout              time.Duration                // The amount of time that a Sandbox can be idle before being terminated.
	Workdir                  string                       // Working directory of the Sandbox.
	Command                  []string                     // Command to run in the Sandbox on startup.
	Env                      map[string]string            // Environment variables to set in the Sandbox.
	Secrets                  []*Secret                    // Secrets to inject into the Sandbox as environment variables.
	Volumes                  map[string]*Volume           // Mount points for Volumes.
	CloudBucketMounts        map[string]*CloudBucketMount // Mount points for cloud buckets.
	PTY                      bool                         // Enable a PTY for the Sandbox entrypoint command. When enabled, all output (stdout and stderr from the process) is multiplexed into stdout, and the stderr stream is effectively empty.
	EncryptedPorts           []int                        // List of encrypted ports to tunnel into the Sandbox, with TLS encryption.
	H2Ports                  []int                        // List of encrypted ports to tunnel into the Sandbox, using HTTP/2.
	UnencryptedPorts         []int                        // List of ports to tunnel into the Sandbox without encryption.
	BlockNetwork             bool                         // Whether to block all network access from the Sandbox.
	CIDRAllowlist            []string                     // List of CIDRs the Sandbox is allowed to access. Cannot be used with BlockNetwork.
	Cloud                    string                       // Cloud provider to run the Sandbox on.
	Regions                  []string                     // Region(s) to run the Sandbox on.
	Verbose                  bool                         // Enable verbose logging.
	Proxy                    *Proxy                       // Reference to a Modal Proxy to use in front of this Sandbox.
	ReadinessProbe           *Probe                       // Probe used to determine when the Sandbox is ready.
	Name                     string                       // Optional name for the Sandbox. Unique within an App.
	ExperimentalOptions      map[string]any               // Experimental options
	CustomDomain             string                       // If non-empty, connections to this Sandbox will be subdomains of this domain rather than the default. This requires prior manual setup by Modal and is only available for Enterprise customers.
	IncludeOidcIdentityToken bool                         // If true, the sandbox will receive a MODAL_IDENTITY_TOKEN env var for OIDC-based auth (e.g. to AWS, GCP).
}

// buildSandboxCreateRequestProto builds a SandboxCreateRequest proto from options.
func buildSandboxCreateRequestProto(appID, imageID string, params SandboxCreateParams) (*pb.SandboxCreateRequest, error) {
	gpuConfig, err := parseGPUConfig(params.GPU)
	if err != nil {
		return nil, err
	}

	if params.Workdir != "" && !strings.HasPrefix(params.Workdir, "/") {
		return nil, fmt.Errorf("the Workdir value must be an absolute path, got: %s", params.Workdir)
	}

	var volumeMounts []*pb.VolumeMount
	if params.Volumes != nil {
		volumeMounts = make([]*pb.VolumeMount, 0, len(params.Volumes))
		for mountPath, volume := range params.Volumes {
			volumeMounts = append(volumeMounts, pb.VolumeMount_builder{
				VolumeId:               volume.VolumeID,
				MountPath:              mountPath,
				AllowBackgroundCommits: true,
				ReadOnly:               volume.IsReadOnly(),
			}.Build())
		}
	}

	var cloudBucketMounts []*pb.CloudBucketMount
	if params.CloudBucketMounts != nil {
		cloudBucketMounts = make([]*pb.CloudBucketMount, 0, len(params.CloudBucketMounts))
		for mountPath, mount := range params.CloudBucketMounts {
			proto, err := mount.toProto(mountPath)
			if err != nil {
				return nil, err
			}
			cloudBucketMounts = append(cloudBucketMounts, proto)
		}
	}

	var ptyInfo *pb.PTYInfo
	if params.PTY {
		ptyInfo = defaultSandboxPTYInfo()
	}

	openPorts := make([]*pb.PortSpec, 0)
	for _, port := range params.EncryptedPorts {
		openPorts = append(openPorts, pb.PortSpec_builder{
			Port:        uint32(port),
			Unencrypted: false,
		}.Build())
	}
	for _, port := range params.H2Ports {
		openPorts = append(openPorts, pb.PortSpec_builder{
			Port:        uint32(port),
			Unencrypted: false,
			TunnelType:  pb.TunnelType_TUNNEL_TYPE_H2.Enum(),
		}.Build())
	}
	for _, port := range params.UnencryptedPorts {
		openPorts = append(openPorts, pb.PortSpec_builder{
			Port:        uint32(port),
			Unencrypted: true,
		}.Build())
	}

	portSpecs := pb.PortSpecs_builder{
		Ports: openPorts,
	}.Build()

	secretIds := []string{}
	for _, secret := range params.Secrets {
		if secret != nil {
			secretIds = append(secretIds, secret.SecretID)
		}
	}

	var networkAccess *pb.NetworkAccess
	if params.BlockNetwork {
		if len(params.CIDRAllowlist) > 0 {
			return nil, fmt.Errorf("CIDRAllowlist cannot be used when BlockNetwork is enabled")
		}
		networkAccess = pb.NetworkAccess_builder{
			NetworkAccessType: pb.NetworkAccess_BLOCKED,
			AllowedCidrs:      []string{},
		}.Build()
	} else if len(params.CIDRAllowlist) > 0 {
		networkAccess = pb.NetworkAccess_builder{
			NetworkAccessType: pb.NetworkAccess_ALLOWLIST,
			AllowedCidrs:      params.CIDRAllowlist,
		}.Build()
	} else {
		networkAccess = pb.NetworkAccess_builder{
			NetworkAccessType: pb.NetworkAccess_OPEN,
			AllowedCidrs:      []string{},
		}.Build()
	}

	var schedulerPlacement *pb.SchedulerPlacement
	if len(params.Regions) > 0 {
		schedulerPlacement = pb.SchedulerPlacement_builder{Regions: params.Regions}.Build()
	}

	var proxyID *string
	if params.Proxy != nil {
		proxyID = &params.Proxy.ProxyID
	}

	var workdir *string
	if params.Workdir != "" {
		workdir = &params.Workdir
	}

	if params.Timeout < 0 {
		return nil, fmt.Errorf("timeout must be non-negative, got %v", params.Timeout)
	}
	if params.Timeout%time.Second != 0 {
		return nil, fmt.Errorf("timeout must be a whole number of seconds, got %v", params.Timeout)
	}
	timeoutSecs := uint32(params.Timeout / time.Second)
	// Ideally we would forbid an explicit zero Timeout, but we can't distinguish between the
	// SandboxCreateParams{Timeout: 0} case that we'd like to warn about, and the SandboxCreateParams{} case
	// where Timeout gets initialized to zero by default.
	// Since Timeout=0 doesn't really make sense, we default to 5 minutes even if it's explicitly set to 0.
	if timeoutSecs == 0 {
		timeoutSecs = 300
	}

	var idleTimeoutSecs *uint32
	if params.IdleTimeout != 0 {
		if params.IdleTimeout < 0 {
			return nil, fmt.Errorf("idleTimeout must be non-negative, got %v", params.IdleTimeout)
		}
		if params.IdleTimeout%time.Second != 0 {
			return nil, fmt.Errorf("idleTimeout must be a whole number of seconds, got %v", params.IdleTimeout)
		}
		v := uint32(params.IdleTimeout / time.Second)
		idleTimeoutSecs = &v
	}

	var milliCPU, milliCPUMax *uint32
	if params.CPU == 0 && params.CPULimit > 0 {
		return nil, fmt.Errorf("must also specify non-zero CPU request when CPULimit is specified")
	}
	if params.CPU != 0 {
		if params.CPU <= 0 {
			return nil, fmt.Errorf("the CPU request (%f) must be a positive number", params.CPU)
		}
		v := uint32(1000 * params.CPU)
		milliCPU = &v
		if params.CPULimit > 0 {
			if params.CPULimit < params.CPU {
				return nil, fmt.Errorf("the CPU request (%f) cannot be higher than CPULimit (%f)", params.CPU, params.CPULimit)
			}
			vMax := uint32(1000 * params.CPULimit)
			milliCPUMax = &vMax
		}
	}

	var memoryMb, memoryMbMax uint32
	if params.MemoryMiB == 0 && params.MemoryLimitMiB > 0 {
		return nil, fmt.Errorf("must also specify non-zero MemoryMiB request when MemoryLimitMiB is specified")
	}
	if params.MemoryMiB != 0 {
		if params.MemoryMiB <= 0 {
			return nil, fmt.Errorf("the MemoryMiB request (%d) must be a positive number", params.MemoryMiB)
		}
		memoryMb = uint32(params.MemoryMiB)
		if params.MemoryLimitMiB > 0 {
			if params.MemoryLimitMiB < params.MemoryMiB {
				return nil, fmt.Errorf("the MemoryMiB request (%d) cannot be higher than MemoryLimitMiB (%d)", params.MemoryMiB, params.MemoryLimitMiB)
			}
			memoryMbMax = uint32(params.MemoryLimitMiB)
		}
	}

	resourcesBuilder := pb.Resources_builder{
		GpuConfig: gpuConfig,
	}
	if milliCPU != nil {
		resourcesBuilder.MilliCpu = *milliCPU
	}
	if milliCPUMax != nil {
		resourcesBuilder.MilliCpuMax = *milliCPUMax
	}
	if memoryMb > 0 {
		resourcesBuilder.MemoryMb = memoryMb
	}
	if memoryMbMax > 0 {
		resourcesBuilder.MemoryMbMax = memoryMbMax
	}

	// The public interface uses map[string]any so that we can add support for any experimental
	// option type in the future. Currently, the proto only supports map[string]bool so we validate
	// the input here.
	protoExperimentalOptions := map[string]bool{}
	for name, value := range params.ExperimentalOptions {
		boolValue, ok := value.(bool)
		if !ok {
			return nil, fmt.Errorf("experimental option '%s' must be a bool, got %T", name, value)
		}
		protoExperimentalOptions[name] = boolValue
	}

	readinessProbe, err := params.ReadinessProbe.toProto()
	if err != nil {
		return nil, err
	}

	return pb.SandboxCreateRequest_builder{
		AppId: appID,
		Definition: pb.Sandbox_builder{
			EntrypointArgs:           params.Command,
			ImageId:                  imageID,
			SecretIds:                secretIds,
			TimeoutSecs:              timeoutSecs,
			IdleTimeoutSecs:          idleTimeoutSecs,
			Workdir:                  workdir,
			NetworkAccess:            networkAccess,
			Resources:                resourcesBuilder.Build(),
			VolumeMounts:             volumeMounts,
			CloudBucketMounts:        cloudBucketMounts,
			PtyInfo:                  ptyInfo,
			OpenPorts:                portSpecs,
			CloudProviderStr:         params.Cloud,
			SchedulerPlacement:       schedulerPlacement,
			Verbose:                  params.Verbose,
			ProxyId:                  proxyID,
			ReadinessProbe:           readinessProbe,
			Name:                     &params.Name,
			ExperimentalOptions:      protoExperimentalOptions,
			CustomDomain:             params.CustomDomain,
			IncludeOidcIdentityToken: params.IncludeOidcIdentityToken,
		}.Build(),
	}.Build(), nil
}

// Create creates a new Sandbox in the App with the specified Image and options.
func (s *sandboxServiceImpl) Create(ctx context.Context, app *App, image *Image, params *SandboxCreateParams) (*Sandbox, error) {
	if params == nil {
		params = &SandboxCreateParams{}
	}

	image, err := image.Build(ctx, app)
	if err != nil {
		return nil, err
	}

	mergedSecrets, err := mergeEnvIntoSecrets(ctx, s.client, &params.Env, &params.Secrets)
	if err != nil {
		return nil, err
	}

	mergedParams := *params
	mergedParams.Secrets = mergedSecrets
	mergedParams.Env = nil // nil'ing Env just to clarify it's not needed anymore

	req, err := buildSandboxCreateRequestProto(app.AppID, image.ImageID, mergedParams)
	if err != nil {
		return nil, err
	}

	createResp, err := s.client.cpClient.SandboxCreate(ctx, req)
	if err != nil {
		if status, ok := status.FromError(err); ok && status.Code() == codes.AlreadyExists {
			return nil, AlreadyExistsError{Exception: status.Message()}
		}
		return nil, err
	}

	s.client.logger.DebugContext(ctx, "Created Sandbox", "sandbox_id", createResp.GetSandboxId())
	return newSandbox(s.client, createResp.GetSandboxId()), nil
}

// StdioBehavior defines how the standard input/output/error streams should behave.
type StdioBehavior string

const (
	// Pipe allows the Sandbox to pipe the streams.
	Pipe StdioBehavior = "pipe"
	// Ignore ignores the streams, meaning they will not be available.
	Ignore StdioBehavior = "ignore"
)

// Tunnel represents a port forwarded from within a running Modal Sandbox.
type Tunnel struct {
	Host            string // The public hostname for the tunnel
	Port            int    // The public port for the tunnel
	UnencryptedHost string // The unencrypted hostname (if applicable)
	UnencryptedPort int    // The unencrypted port (if applicable)
}

// URL gets the public HTTPS URL of the forwarded port.
func (t *Tunnel) URL() string {
	if t.Port == 443 {
		return fmt.Sprintf("https://%s", t.Host)
	}
	return fmt.Sprintf("https://%s:%d", t.Host, t.Port)
}

// TLSSocket gets the public TLS socket as a (host, port) tuple.
func (t *Tunnel) TLSSocket() (string, int) {
	return t.Host, t.Port
}

// TCPSocket gets the public TCP socket as a (host, port) tuple.
func (t *Tunnel) TCPSocket() (string, int, error) {
	if t.UnencryptedHost == "" || t.UnencryptedPort == 0 {
		return "", 0, InvalidError{Exception: "This tunnel is not configured for unencrypted TCP."}
	}
	return t.UnencryptedHost, t.UnencryptedPort, nil
}

// Sandbox represents a Modal Sandbox, which can run commands and manage
// input/output streams for a remote process. After you are done interacting with the sandbox,
// we recommend calling [Sandbox.Detach] which disconnects your client from the sandbox and
// cleans up any resources associated with the connection.
type Sandbox struct {
	SandboxID string
	Stdin     io.WriteCloser
	Stdout    io.ReadCloser
	Stderr    io.ReadCloser

	taskID  string
	tunnels map[int]*Tunnel

	client *Client

	commandRouterClient   *taskCommandRouterClient
	commandRouterClientMu sync.Mutex

	attached atomic.Bool
}

func defaultSandboxPTYInfo() *pb.PTYInfo {
	return pb.PTYInfo_builder{
		Enabled:                true,
		WinszRows:              24,
		WinszCols:              80,
		EnvTerm:                "xterm-256color",
		EnvColorterm:           "truecolor",
		PtyType:                pb.PTYInfo_PTY_TYPE_SHELL,
		NoTerminateOnIdleStdin: true,
	}.Build()
}

// newSandbox creates a new Sandbox object from ID.
func newSandbox(client *Client, sandboxID string) *Sandbox {
	sb := &Sandbox{SandboxID: sandboxID, client: client}
	sb.attached.Store(true)
	sb.Stdin = inputStreamSb(client.cpClient, sandboxID)
	sb.Stdout = &lazyStreamReader{
		initFunc: func() io.ReadCloser {
			return outputStreamSb(client.cpClient, client.logger, sandboxID, pb.FileDescriptor_FILE_DESCRIPTOR_STDOUT)
		},
	}
	sb.Stderr = &lazyStreamReader{
		initFunc: func() io.ReadCloser {
			return outputStreamSb(client.cpClient, client.logger, sandboxID, pb.FileDescriptor_FILE_DESCRIPTOR_STDERR)
		},
	}
	return sb
}

// FromID returns a running Sandbox object from an ID.
func (s *sandboxServiceImpl) FromID(ctx context.Context, sandboxID string) (*Sandbox, error) {
	_, err := s.client.cpClient.SandboxWait(ctx, pb.SandboxWaitRequest_builder{
		SandboxId: sandboxID,
		Timeout:   0,
	}.Build())
	if status, ok := status.FromError(err); ok && status.Code() == codes.NotFound {
		return nil, NotFoundError{fmt.Sprintf("Sandbox with id: '%s' not found", sandboxID)}
	}
	if err != nil {
		return nil, err
	}
	return newSandbox(s.client, sandboxID), nil
}

// SandboxFromNameParams are options for finding deployed Sandbox objects by name.
type SandboxFromNameParams struct {
	Environment string
}

// FromName gets a running Sandbox by name from a deployed App.
//
// Raises a NotFoundError if no running Sandbox is found with the given name.
// A Sandbox's name is the `Name` argument passed to `App.CreateSandbox`.
func (s *sandboxServiceImpl) FromName(ctx context.Context, appName, name string, params *SandboxFromNameParams) (*Sandbox, error) {
	if params == nil {
		params = &SandboxFromNameParams{}
	}

	resp, err := s.client.cpClient.SandboxGetFromName(ctx, pb.SandboxGetFromNameRequest_builder{
		SandboxName:     name,
		AppName:         appName,
		EnvironmentName: environmentName(params.Environment, s.client.profile),
	}.Build())
	if err != nil {
		if status, ok := status.FromError(err); ok && status.Code() == codes.NotFound {
			return nil, NotFoundError{Exception: fmt.Sprintf("Sandbox with name '%s' not found in pp '%s'", name, appName)}
		}
		return nil, err
	}

	return newSandbox(s.client, resp.GetSandboxId()), nil
}

// SandboxExecParams defines options for executing commands in a Sandbox.
type SandboxExecParams struct {
	// Stdout defines whether to pipe or ignore standard output.
	Stdout StdioBehavior
	// Stderr defines whether to pipe or ignore standard error.
	Stderr StdioBehavior
	// Workdir is the working directory to run the command in.
	Workdir string
	// Timeout is the timeout for command execution. Defaults to 0 (no timeout).
	Timeout time.Duration
	// Environment variables to set for the command.
	Env map[string]string
	// Secrets to inject as environment variables for the command.
	Secrets []*Secret
	// PTY defines whether to enable a PTY for the command. When enabled, all output (stdout and
	// stderr from the process) is multiplexed into stdout, and the stderr stream is effectively empty.
	PTY bool
}

// ValidateExecArgs checks if command arguments exceed ARG_MAX.
func ValidateExecArgs(args []string) error {
	// The maximum number of bytes that can be passed to an exec on Linux.
	// Though this is technically a 'server side' limit, it is unlikely to change.
	// getconf ARG_MAX will show this value on a host.
	//
	// By probing in production, the limit is 131072 bytes (2**17).
	// We need some bytes of overhead for the rest of the command line besides the args,
	// e.g. 'runsc exec ...'. So we use 2**16 as the limit.

	argMaxBytes := 1 << 16

	// Avoid "[Errno 7] Argument list too long" errors.
	totalLen := 0
	for _, arg := range args {
		totalLen += len(arg)
	}
	if totalLen > argMaxBytes {
		return InvalidError{Exception: fmt.Sprintf(
			"Total length of CMD arguments must be less than %d bytes. Got %d bytes.",
			argMaxBytes, totalLen,
		)}
	}
	return nil
}

// buildTaskExecStartRequestProto builds a TaskExecStartRequest proto from command and options.
func buildTaskExecStartRequestProto(taskID, execID string, command []string, params SandboxExecParams) (*pb.TaskExecStartRequest, error) {
	if params.Timeout < 0 {
		return nil, fmt.Errorf("timeout must be non-negative, got %v", params.Timeout)
	}
	if params.Timeout != 0 && params.Timeout%time.Second != 0 {
		return nil, fmt.Errorf("timeout must be a whole number of seconds, got %v", params.Timeout)
	}

	secretIds := []string{}
	for _, secret := range params.Secrets {
		if secret != nil {
			secretIds = append(secretIds, secret.SecretID)
		}
	}

	var stdoutConfig pb.TaskExecStdoutConfig
	switch params.Stdout {
	case Pipe, "":
		stdoutConfig = pb.TaskExecStdoutConfig_TASK_EXEC_STDOUT_CONFIG_PIPE
	case Ignore:
		stdoutConfig = pb.TaskExecStdoutConfig_TASK_EXEC_STDOUT_CONFIG_DEVNULL
	default:
		return nil, fmt.Errorf("unsupported stdout behavior: %s", params.Stdout)
	}

	var stderrConfig pb.TaskExecStderrConfig
	switch params.Stderr {
	case Pipe, "":
		stderrConfig = pb.TaskExecStderrConfig_TASK_EXEC_STDERR_CONFIG_PIPE
	case Ignore:
		stderrConfig = pb.TaskExecStderrConfig_TASK_EXEC_STDERR_CONFIG_DEVNULL
	default:
		return nil, fmt.Errorf("unsupported stderr behavior: %s", params.Stderr)
	}

	var ptyInfo *pb.PTYInfo
	if params.PTY {
		ptyInfo = defaultSandboxPTYInfo()
	}

	builder := pb.TaskExecStartRequest_builder{
		TaskId:       taskID,
		ExecId:       execID,
		CommandArgs:  command,
		StdoutConfig: stdoutConfig,
		StderrConfig: stderrConfig,
		Workdir:      nil,
		SecretIds:    secretIds,
		PtyInfo:      ptyInfo,
		RuntimeDebug: false,
	}

	if params.Workdir != "" {
		builder.Workdir = &params.Workdir
	}

	if params.Timeout > 0 {
		timeoutSecs := uint32(params.Timeout / time.Second)
		builder.TimeoutSecs = &timeoutSecs
	}

	return builder.Build(), nil
}

// Exec runs a command in the Sandbox and returns text streams.
func (sb *Sandbox) Exec(ctx context.Context, command []string, params *SandboxExecParams) (*ContainerProcess, error) {
	if err := sb.ensureAttached(); err != nil {
		return nil, err
	}

	if params == nil {
		params = &SandboxExecParams{}
	}

	if err := ValidateExecArgs(command); err != nil {
		return nil, err
	}

	if err := sb.ensureTaskID(ctx); err != nil {
		return nil, err
	}

	mergedSecrets, err := mergeEnvIntoSecrets(ctx, sb.client, &params.Env, &params.Secrets)
	if err != nil {
		return nil, err
	}

	mergedParams := *params
	mergedParams.Secrets = mergedSecrets
	mergedParams.Env = nil // nil'ing Env just to clarify it's not needed anymore

	commandRouterClient, err := sb.getOrCreateCommandRouterClient(ctx, sb.taskID)
	if err != nil {
		return nil, err
	}

	execID := uuid.New().String()
	req, err := buildTaskExecStartRequestProto(sb.taskID, execID, command, mergedParams)
	if err != nil {
		return nil, err
	}

	_, err = commandRouterClient.ExecStart(ctx, req)
	if err != nil {
		return nil, err
	}

	sb.client.logger.DebugContext(ctx, "Created ContainerProcess",
		"exec_id", execID,
		"sandbox_id", sb.SandboxID,
		"command", command)

	var deadline *time.Time
	if mergedParams.Timeout > 0 {
		d := time.Now().Add(mergedParams.Timeout)
		deadline = &d
	}

	return newContainerProcess(commandRouterClient, sb.client.logger, sb.taskID, execID, mergedParams, deadline), nil
}

// SandboxCreateConnectTokenParams are optional parameters for CreateConnectToken.
type SandboxCreateConnectTokenParams struct {
	// Optional user-provided metadata string that will be added to the headers by the proxy when forwarding requests to the Sandbox.
	UserMetadata string
}

// SandboxCreateConnectCredentials contains the URL and token for connecting to a Sandbox.
type SandboxCreateConnectCredentials struct {
	URL   string
	Token string
}

// CreateConnectToken creates a token for making HTTP connections to the Sandbox.
func (sb *Sandbox) CreateConnectToken(ctx context.Context, params *SandboxCreateConnectTokenParams) (*SandboxCreateConnectCredentials, error) {
	if err := sb.ensureAttached(); err != nil {
		return nil, err
	}

	if params == nil {
		params = &SandboxCreateConnectTokenParams{}
	}
	resp, err := sb.client.cpClient.SandboxCreateConnectToken(ctx, pb.SandboxCreateConnectTokenRequest_builder{
		SandboxId:    sb.SandboxID,
		UserMetadata: params.UserMetadata,
	}.Build())
	if err != nil {
		return nil, err
	}
	return &SandboxCreateConnectCredentials{URL: resp.GetUrl(), Token: resp.GetToken()}, nil
}

// Open opens a file in the Sandbox filesystem.
// The mode parameter follows the same conventions as os.OpenFile:
// "r" for read-only, "w" for write-only (truncates), "a" for append, etc.
func (sb *Sandbox) Open(ctx context.Context, path, mode string) (*SandboxFile, error) {
	if err := sb.ensureAttached(); err != nil {
		return nil, err
	}

	if err := sb.ensureTaskID(ctx); err != nil {
		return nil, err
	}

	_, resp, err := runFilesystemExec(ctx, sb.client.cpClient, pb.ContainerFilesystemExecRequest_builder{
		FileOpenRequest: pb.ContainerFileOpenRequest_builder{
			Path: path,
			Mode: mode,
		}.Build(),
		TaskId: sb.taskID,
	}.Build(), nil)

	if err != nil {
		return nil, err
	}

	return &SandboxFile{
		fileDescriptor: resp.GetFileDescriptor(),
		taskID:         sb.taskID,
		cpClient:       sb.client.cpClient,
	}, nil
}

const maxGetTaskIDAttempts = 600 // 5 minutes at 500ms intervals

func (sb *Sandbox) ensureTaskID(ctx context.Context) error {
	if sb.taskID != "" {
		return nil
	}
	for range maxGetTaskIDAttempts {
		resp, err := sb.client.cpClient.SandboxGetTaskId(ctx, pb.SandboxGetTaskIdRequest_builder{
			SandboxId: sb.SandboxID,
		}.Build())
		if err != nil {
			return err
		}
		if resp.GetTaskResult() != nil {
			return fmt.Errorf("Sandbox %s has already completed with result: %v", sb.SandboxID, resp.GetTaskResult())
		}
		if resp.GetTaskId() != "" {
			sb.taskID = resp.GetTaskId()
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(500 * time.Millisecond):
		}
	}
	return fmt.Errorf("timed out waiting for task ID for Sandbox %s", sb.SandboxID)
}

func (sb *Sandbox) getOrCreateCommandRouterClient(ctx context.Context, taskID string) (*taskCommandRouterClient, error) {
	sb.commandRouterClientMu.Lock()
	defer sb.commandRouterClientMu.Unlock()

	if err := sb.ensureAttached(); err != nil {
		return nil, err
	}

	if sb.commandRouterClient == nil {
		client, err := initTaskCommandRouterClient(
			ctx,
			sb.client.cpClient,
			taskID,
			sb.client.logger,
			sb.client.profile,
		)
		if err != nil {
			return nil, err
		}
		sb.commandRouterClient = client
	}
	return sb.commandRouterClient, nil
}
func (sb *Sandbox) ensureAttached() error {
	if !sb.attached.Load() {
		return ClientClosedError{Exception: "Unable to perform operation on a detached sandbox"}
	}
	return nil
}

// Detach disconnects from the running Sandbox
func (sb *Sandbox) Detach() error {
	if !sb.attached.Load() {
		return nil
	}
	sb.commandRouterClientMu.Lock()
	defer sb.commandRouterClientMu.Unlock()

	if sb.commandRouterClient != nil {
		err := sb.commandRouterClient.Close()
		if err != nil {
			return err
		}
		sb.commandRouterClient = nil
	}
	sb.attached.CompareAndSwap(true, false)
	return nil
}

// SandboxTerminateParams are options for Terminate. If Wait is true, then `Terminate`
// will wait for the Sandbox to terminate and return the exit code.
type SandboxTerminateParams struct {
	Wait bool
}

// Terminate stops the Sandbox.
func (sb *Sandbox) Terminate(ctx context.Context, params *SandboxTerminateParams) (int, error) {
	if err := sb.ensureAttached(); err != nil {
		return 0, err
	}

	if params == nil {
		params = &SandboxTerminateParams{}
	}

	// Terminate the sandbox even if detach fails.
	_, err := sb.client.cpClient.SandboxTerminate(ctx, pb.SandboxTerminateRequest_builder{
		SandboxId: sb.SandboxID,
	}.Build())
	if err != nil {
		return 0, err
	}
	sb.taskID = ""
	returnCode := 0

	if params.Wait {
		returnCode, err = sb.Wait(ctx)
		// If Wait fails, we do not detach yet
		if err != nil {
			return returnCode, err
		}
	}

	err = sb.Detach()
	if err != nil {
		return returnCode, err
	}

	return returnCode, nil
}

// Wait blocks until the Sandbox exits.
func (sb *Sandbox) Wait(ctx context.Context) (int, error) {
	for {
		if err := ctx.Err(); err != nil {
			return 0, err
		}

		resp, err := sb.client.cpClient.SandboxWait(ctx, pb.SandboxWaitRequest_builder{
			SandboxId: sb.SandboxID,
			Timeout:   10,
		}.Build())
		if err != nil {
			return 0, err
		}
		if resp.GetResult() != nil {
			returnCode := getReturnCode(resp.GetResult())
			sb.client.logger.DebugContext(ctx, "Sandbox wait completed",
				"sandbox_id", sb.SandboxID,
				"status", resp.GetResult().GetStatus().String(),
				"return_code", returnCode)
			if returnCode != nil {
				return *returnCode, nil
			}
			return 0, nil
		}
	}
}

// WaitUntilReady blocks until the Sandbox readiness probe reports ready.
func (sb *Sandbox) WaitUntilReady(ctx context.Context, timeout time.Duration) error {
	if err := sb.ensureAttached(); err != nil {
		return err
	}
	if timeout <= 0 {
		return InvalidError{Exception: fmt.Sprintf("timeout must be positive, got %v", timeout)}
	}

	deadline := time.Now().Add(timeout)
	for {
		if err := ctx.Err(); err != nil {
			return err
		}

		remaining := time.Until(deadline)
		if remaining <= 0 {
			return TimeoutError{Exception: "Sandbox operation timed out"}
		}

		requestTimeout := min(
			remaining,
			50*time.Second, // Max timeout for a single gRPC call.
		)

		resp, err := sb.client.cpClient.SandboxWaitUntilReady(ctx, pb.SandboxWaitUntilReadyRequest_builder{
			SandboxId: sb.SandboxID,
			Timeout:   float32(requestTimeout.Seconds()),
		}.Build())
		if err != nil {
			if status, ok := status.FromError(err); ok && status.Code() == codes.DeadlineExceeded {
				continue
			}
			return err
		}
		if resp.GetReadyAt() > 0 {
			return nil
		}
	}
}

// Tunnels gets Tunnel metadata for the Sandbox.
// Returns SandboxTimeoutError if the tunnels are not available after the timeout.
// Returns a map of Tunnel objects keyed by the container port.
func (sb *Sandbox) Tunnels(ctx context.Context, timeout time.Duration) (map[int]*Tunnel, error) {
	if err := sb.ensureAttached(); err != nil {
		return nil, err
	}
	if sb.tunnels != nil {
		return sb.tunnels, nil
	}

	resp, err := sb.client.cpClient.SandboxGetTunnels(ctx, pb.SandboxGetTunnelsRequest_builder{
		SandboxId: sb.SandboxID,
		Timeout:   float32(timeout.Seconds()),
	}.Build())
	if err != nil {
		return nil, err
	}

	if resp.GetResult() != nil && resp.GetResult().GetStatus() == pb.GenericResult_GENERIC_STATUS_TIMEOUT {
		return nil, SandboxTimeoutError{Exception: "Sandbox operation timed out"}
	}

	sb.tunnels = make(map[int]*Tunnel)
	for _, t := range resp.GetTunnels() {
		sb.tunnels[int(t.GetContainerPort())] = &Tunnel{
			Host:            t.GetHost(),
			Port:            int(t.GetPort()),
			UnencryptedHost: t.GetUnencryptedHost(),
			UnencryptedPort: int(t.GetUnencryptedPort()),
		}
	}

	return sb.tunnels, nil
}

// SnapshotFilesystem takes a snapshot of the Sandbox's filesystem.
// Returns an Image object which can be used to spawn a new Sandbox with the same filesystem.
func (sb *Sandbox) SnapshotFilesystem(ctx context.Context, timeout time.Duration) (*Image, error) {
	if err := sb.ensureAttached(); err != nil {
		return nil, err
	}
	resp, err := sb.client.cpClient.SandboxSnapshotFs(ctx, pb.SandboxSnapshotFsRequest_builder{
		SandboxId: sb.SandboxID,
		Timeout:   float32(timeout.Seconds()),
	}.Build())
	if err != nil {
		return nil, err
	}

	if resp.GetResult() != nil && resp.GetResult().GetStatus() != pb.GenericResult_GENERIC_STATUS_SUCCESS {
		return nil, ExecutionError{Exception: fmt.Sprintf("Sandbox snapshot failed: %s", resp.GetResult().GetException())}
	}

	if resp.GetImageId() == "" {
		return nil, ExecutionError{Exception: "Sandbox snapshot response missing image ID"}
	}

	return &Image{ImageID: resp.GetImageId(), client: sb.client}, nil
}

// MountImage mounts an Image at a path in the Sandbox filesystem.
//
// If image is nil, mounts an empty directory.
func (sb *Sandbox) MountImage(ctx context.Context, path string, image *Image) error {
	if err := sb.ensureAttached(); err != nil {
		return err
	}
	if err := sb.ensureTaskID(ctx); err != nil {
		return err
	}

	crClient, err := sb.getOrCreateCommandRouterClient(ctx, sb.taskID)
	if err != nil {
		return err
	}

	imageID := ""
	if image != nil {
		if image.ImageID == "" {
			return InvalidError{Exception: "Image must be built before mounting. Call `image.Build(app)` first."}
		}
		imageID = image.ImageID
	}

	request := pb.TaskMountDirectoryRequest_builder{
		TaskId:  sb.taskID,
		Path:    []byte(path),
		ImageId: imageID,
	}.Build()

	return crClient.MountDirectory(ctx, request)
}

// SnapshotDirectory snapshots and creates a new image from a directory in the running sandbox.
func (sb *Sandbox) SnapshotDirectory(ctx context.Context, path string) (*Image, error) {
	if err := sb.ensureAttached(); err != nil {
		return nil, err
	}
	if err := sb.ensureTaskID(ctx); err != nil {
		return nil, err
	}

	crClient, err := sb.getOrCreateCommandRouterClient(ctx, sb.taskID)
	if err != nil {
		return nil, err
	}

	request := pb.TaskSnapshotDirectoryRequest_builder{
		TaskId: sb.taskID,
		Path:   []byte(path),
	}.Build()

	response, err := crClient.SnapshotDirectory(ctx, request)
	if err != nil {
		return nil, err
	}

	if response.GetImageId() == "" {
		return nil, ExecutionError{Exception: "Sandbox snapshot directory response missing `imageId`"}
	}

	return &Image{ImageID: response.GetImageId(), client: sb.client}, nil
}

// Poll checks if the Sandbox has finished running.
// Returns nil if the Sandbox is still running, else returns the exit code.
func (sb *Sandbox) Poll(ctx context.Context) (*int, error) {
	if err := sb.ensureAttached(); err != nil {
		return nil, err
	}
	resp, err := sb.client.cpClient.SandboxWait(ctx, pb.SandboxWaitRequest_builder{
		SandboxId: sb.SandboxID,
		Timeout:   0,
	}.Build())
	if err != nil {
		return nil, err
	}

	return getReturnCode(resp.GetResult()), nil
}

// SetTags sets key-value tags on the Sandbox. Tags can be used to filter results in SandboxList.
func (sb *Sandbox) SetTags(ctx context.Context, tags map[string]string) error {
	if err := sb.ensureAttached(); err != nil {
		return err
	}
	tagsList := make([]*pb.SandboxTag, 0, len(tags))
	for k, v := range tags {
		tagsList = append(tagsList, pb.SandboxTag_builder{TagName: k, TagValue: v}.Build())
	}
	_, err := sb.client.cpClient.SandboxTagsSet(ctx, pb.SandboxTagsSetRequest_builder{
		EnvironmentName: environmentName("", sb.client.profile),
		SandboxId:       sb.SandboxID,
		Tags:            tagsList,
	}.Build())
	return err
}

// GetTags fetches any tags (key-value pairs) currently attached to this Sandbox from the server.
func (sb *Sandbox) GetTags(ctx context.Context) (map[string]string, error) {
	if err := sb.ensureAttached(); err != nil {
		return nil, err
	}
	resp, err := sb.client.cpClient.SandboxTagsGet(ctx, pb.SandboxTagsGetRequest_builder{
		SandboxId: sb.SandboxID,
	}.Build())
	if err != nil {
		if status, ok := status.FromError(err); ok && status.Code() == codes.InvalidArgument {
			return nil, InvalidError{Exception: status.Message()}
		}
		return nil, err
	}

	tags := make(map[string]string, len(resp.GetTags()))
	for _, tag := range resp.GetTags() {
		tags[tag.GetTagName()] = tag.GetTagValue()
	}
	return tags, nil
}

// SandboxListParams are options for listing Sandboxes.
type SandboxListParams struct {
	AppID       string            // Filter by App ID
	Tags        map[string]string // Only include Sandboxes that have all these tags
	Environment string            // Override environment for this request
}

// List lists Sandboxes for the current environment (or provided App ID), optionally filtered by tags.
func (s *sandboxServiceImpl) List(ctx context.Context, params *SandboxListParams) (iter.Seq2[*Sandbox, error], error) {
	if params == nil {
		params = &SandboxListParams{}
	}

	tagsList := make([]*pb.SandboxTag, 0, len(params.Tags))
	for k, v := range params.Tags {
		tagsList = append(tagsList, pb.SandboxTag_builder{TagName: k, TagValue: v}.Build())
	}

	return func(yield func(*Sandbox, error) bool) {
		var before float64
		for {
			if err := ctx.Err(); err != nil {
				yield(nil, err)
				return
			}

			resp, err := s.client.cpClient.SandboxList(ctx, pb.SandboxListRequest_builder{
				AppId:           params.AppID,
				BeforeTimestamp: before,
				EnvironmentName: environmentName(params.Environment, s.client.profile),
				IncludeFinished: false,
				Tags:            tagsList,
			}.Build())
			if err != nil {
				yield(nil, err)
				return
			}
			sandboxes := resp.GetSandboxes()
			if len(sandboxes) == 0 {
				return
			}
			for _, info := range sandboxes {
				if !yield(newSandbox(s.client, info.GetId()), nil) {
					return
				}
			}
			before = sandboxes[len(sandboxes)-1].GetCreatedAt()
		}
	}, nil
}

func getReturnCode(result *pb.GenericResult) *int {
	if result == nil || result.GetStatus() == pb.GenericResult_GENERIC_STATUS_UNSPECIFIED {
		return nil
	}

	// Statuses are converted to exitcodes so we can conform to subprocess API.
	var exitCode int
	switch result.GetStatus() {
	case pb.GenericResult_GENERIC_STATUS_TIMEOUT:
		exitCode = 124
	case pb.GenericResult_GENERIC_STATUS_TERMINATED:
		exitCode = 137
	default:
		exitCode = int(result.GetExitcode())
	}

	return &exitCode
}

// ContainerProcess represents a process running in a Modal container, allowing
// interaction with its standard input/output/error streams.
//
// It is created by executing a command in a Sandbox.
type ContainerProcess struct {
	Stdin  io.WriteCloser
	Stdout io.ReadCloser
	Stderr io.ReadCloser

	taskID              string
	execID              string
	commandRouterClient *taskCommandRouterClient
	deadline            *time.Time
}

func newContainerProcess(commandRouterClient *taskCommandRouterClient, logger *slog.Logger, taskID, execID string, params SandboxExecParams, deadline *time.Time) *ContainerProcess {
	stdoutBehavior := Pipe
	stderrBehavior := Pipe
	if params.Stdout != "" {
		stdoutBehavior = params.Stdout
	}
	if params.Stderr != "" {
		stderrBehavior = params.Stderr
	}

	cp := &ContainerProcess{
		taskID:              taskID,
		execID:              execID,
		commandRouterClient: commandRouterClient,
		deadline:            deadline,
	}
	cp.Stdin = inputStreamCp(commandRouterClient, taskID, execID)

	if stdoutBehavior == Ignore {
		cp.Stdout = io.NopCloser(bytes.NewReader(nil))
	} else {
		cp.Stdout = &lazyStreamReader{
			initFunc: func() io.ReadCloser {
				return outputStreamCp(commandRouterClient, logger, taskID, execID, pb.FileDescriptor_FILE_DESCRIPTOR_STDOUT, deadline)
			},
		}
	}
	if stderrBehavior == Ignore {
		cp.Stderr = io.NopCloser(bytes.NewReader(nil))
	} else {
		cp.Stderr = &lazyStreamReader{
			initFunc: func() io.ReadCloser {
				return outputStreamCp(commandRouterClient, logger, taskID, execID, pb.FileDescriptor_FILE_DESCRIPTOR_STDERR, deadline)
			},
		}
	}

	return cp
}

// Wait blocks until the container process exits and returns its exit code.
func (cp *ContainerProcess) Wait(ctx context.Context) (int, error) {
	resp, err := cp.commandRouterClient.ExecWait(ctx, cp.taskID, cp.execID, cp.deadline)
	if err != nil {
		return 0, err
	}
	switch resp.WhichExitStatus() {
	case pb.TaskExecWaitResponse_Code_case:
		return int(resp.GetCode()), nil
	case pb.TaskExecWaitResponse_Signal_case:
		return 128 + int(resp.GetSignal()), nil
	default:
		return 0, InvalidError{Exception: "Unexpected exit status"}
	}
}

func inputStreamSb(cpClient pb.ModalClientClient, sandboxID string) io.WriteCloser {
	return &sbStdin{sandboxID: sandboxID, index: 1, cpClient: cpClient}
}

type sbStdin struct {
	sandboxID string
	cpClient  pb.ModalClientClient

	mu    sync.Mutex // protects index
	index uint32
}

func (sbs *sbStdin) Write(p []byte) (int, error) {
	sbs.mu.Lock()
	defer sbs.mu.Unlock()
	index := sbs.index
	sbs.index++
	_, err := sbs.cpClient.SandboxStdinWrite(context.Background(), pb.SandboxStdinWriteRequest_builder{
		SandboxId: sbs.sandboxID,
		Input:     p,
		Index:     index,
	}.Build())
	if err != nil {
		return 0, err
	}
	return len(p), nil
}

func (sbs *sbStdin) Close() error {
	sbs.mu.Lock()
	defer sbs.mu.Unlock()
	_, err := sbs.cpClient.SandboxStdinWrite(context.Background(), pb.SandboxStdinWriteRequest_builder{
		SandboxId: sbs.sandboxID,
		Index:     sbs.index,
		Eof:       true,
	}.Build())
	if st, ok := status.FromError(err); ok && st.Code() == codes.FailedPrecondition {
		return nil
	}
	return err
}

func inputStreamCp(commandRouterClient *taskCommandRouterClient, taskID, execID string) io.WriteCloser {
	return &cpStdin{taskID: taskID, execID: execID, offset: 0, commandRouterClient: commandRouterClient}
}

type cpStdin struct {
	taskID              string
	execID              string
	commandRouterClient *taskCommandRouterClient
	offset              uint64
}

func (cps *cpStdin) Write(p []byte) (int, error) {
	err := cps.commandRouterClient.ExecStdinWrite(context.Background(), cps.taskID, cps.execID, cps.offset, p, false)
	if err != nil {
		return 0, err
	}
	cps.offset += uint64(len(p))
	return len(p), nil
}

func (cps *cpStdin) Close() error {
	return cps.commandRouterClient.ExecStdinWrite(context.Background(), cps.taskID, cps.execID, cps.offset, nil, true)
}

// cancelOnCloseReader is used to cancel background goroutines when the stream is closed.
type cancelOnCloseReader struct {
	io.ReadCloser
	cancel context.CancelFunc
}

func (r *cancelOnCloseReader) Close() error {
	r.cancel()
	return r.ReadCloser.Close()
}

// lazyStreamReader defers stream initialization until the first read, preventing goroutine
// leaks for unused streams. Without lazy initialization, output stream goroutines are created
// eagerly and block on stream.Recv() calls.
type lazyStreamReader struct {
	once     sync.Once
	reader   io.ReadCloser
	initFunc func() io.ReadCloser
}

func (l *lazyStreamReader) Read(p []byte) (int, error) {
	l.once.Do(func() {
		l.reader = l.initFunc()
	})
	return l.reader.Read(p)
}

func (l *lazyStreamReader) Close() error {
	l.once.Do(func() {
		l.reader = io.NopCloser(bytes.NewReader(nil))
	})
	if l.reader != nil {
		return l.reader.Close()
	}
	return nil
}

func outputStreamSb(cpClient pb.ModalClientClient, logger *slog.Logger, sandboxID string, fd pb.FileDescriptor) io.ReadCloser {
	pr, pw := nio.Pipe(buffer.New(64 * 1024))
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		defer func() {
			if err := pw.Close(); err != nil {
				logger.DebugContext(ctx, "failed to close pipe writer", "error", err.Error())
			}
		}()
		defer cancel()
		lastIndex := "0-0"
		completed := false
		retries := 10
		for !completed {
			stream, err := cpClient.SandboxGetLogs(ctx, pb.SandboxGetLogsRequest_builder{
				SandboxId:      sandboxID,
				FileDescriptor: fd,
				Timeout:        55,
				LastEntryId:    lastIndex,
			}.Build())
			if err != nil {
				if ctx.Err() != nil {
					return
				}
				if isRetryableGrpc(err) && retries > 0 {
					retries--
					continue
				}
				streamErr := fmt.Errorf("error getting output stream: %w", err)
				if closeErr := pw.CloseWithError(streamErr); closeErr != nil {
					logger.DebugContext(ctx, "failed to close pipe writer with error", "error", closeErr.Error(), "stream_error", streamErr.Error())
				}
				return
			}
			for {
				batch, err := stream.Recv()
				if err != nil {
					if ctx.Err() != nil {
						return
					}
					if err != io.EOF {
						if isRetryableGrpc(err) && retries > 0 {
							retries--
						} else {
							streamErr := fmt.Errorf("error getting output stream: %w", err)
							if closeErr := pw.CloseWithError(streamErr); closeErr != nil {
								logger.DebugContext(ctx, "failed to close pipe writer with error", "error", closeErr.Error(), "stream_error", streamErr.Error())
							}
							return
						}
					}
					break // we need to retry, either from an EOF or gRPC error
				}
				lastIndex = batch.GetEntryId()
				for _, item := range batch.GetItems() {
					// On error, writer has been closed. Still consume the rest of the channel.
					if _, err := pw.Write([]byte(item.GetData())); err != nil {
						logger.DebugContext(ctx, "failed to write to pipe", "error", err.Error())
					}
				}
				if batch.GetEof() {
					completed = true
					break
				}
			}
		}
	}()
	return &cancelOnCloseReader{ReadCloser: pr, cancel: cancel}
}

func outputStreamCp(commandRouterClient *taskCommandRouterClient, logger *slog.Logger, taskID, execID string, fd pb.FileDescriptor, deadline *time.Time) io.ReadCloser {
	pr, pw := nio.Pipe(buffer.New(64 * 1024))
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		defer func() {
			if err := pw.Close(); err != nil {
				logger.DebugContext(ctx, "failed to close pipe writer", "error", err.Error())
			}
		}()
		defer cancel()

		resultCh := commandRouterClient.ExecStdioRead(ctx, taskID, execID, fd, deadline)
		for result := range resultCh {
			if result.Err != nil {
				if ctx.Err() != nil {
					return
				}
				streamErr := fmt.Errorf("error getting output stream: %w", result.Err)
				if closeErr := pw.CloseWithError(streamErr); closeErr != nil {
					logger.DebugContext(ctx, "failed to close pipe writer with error", "error", closeErr.Error(), "stream_error", streamErr.Error())
				}
				return
			}
			if _, err := pw.Write(result.Response.GetData()); err != nil {
				logger.DebugContext(ctx, "failed to write to pipe", "error", err.Error())
			}
		}
	}()
	return &cancelOnCloseReader{ReadCloser: pr, cancel: cancel}
}
