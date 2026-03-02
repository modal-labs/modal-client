package modal

// errors.go defines common error types for the public API.

// FunctionTimeoutError is returned when a Function execution exceeds the allowed time limit.
type FunctionTimeoutError struct {
	Exception string
}

func (e FunctionTimeoutError) Error() string {
	return "FunctionTimeoutError: " + e.Exception
}

// RemoteError represents an error on the Modal server, or a Python exception.
type RemoteError struct {
	Exception string
}

func (e RemoteError) Error() string {
	return "RemoteError: " + e.Exception
}

// InternalFailure is a retryable internal error from Modal.
type InternalFailure struct {
	Exception string
}

func (e InternalFailure) Error() string {
	return "InternalFailure: " + e.Exception
}

// ExecutionError is returned when something unexpected happened during runtime.
type ExecutionError struct {
	Exception string
}

func (e ExecutionError) Error() string {
	return "ExecutionError: " + e.Exception
}

// NotFoundError is returned when a resource is not found.
type NotFoundError struct {
	Exception string
}

func (e NotFoundError) Error() string {
	return "NotFoundError: " + e.Exception
}

// AlreadyExistsError is returned when a resource already exists.
type AlreadyExistsError struct {
	Exception string
}

func (e AlreadyExistsError) Error() string {
	return "AlreadyExistsError: " + e.Exception
}

// InvalidError represents an invalid request or operation.
type InvalidError struct {
	Exception string
}

func (e InvalidError) Error() string {
	return "InvalidError: " + e.Exception
}

// QueueEmptyError is returned when an operation is attempted on an empty Queue.
type QueueEmptyError struct {
	Exception string
}

func (e QueueEmptyError) Error() string {
	return "QueueEmptyError: " + e.Exception
}

// QueueFullError is returned when an operation is attempted on a full Queue.
type QueueFullError struct {
	Exception string
}

func (e QueueFullError) Error() string {
	return "QueueFullError: " + e.Exception
}

// SandboxFilesystemError is returned when an operation is attempted on a full Queue.
type SandboxFilesystemError struct {
	Exception string
}

func (e SandboxFilesystemError) Error() string {
	return "SandboxFilesystemError: " + e.Exception
}

// SandboxTimeoutError is returned when Sandbox operations exceed the allowed time limit.
type SandboxTimeoutError struct {
	Exception string
}

func (e SandboxTimeoutError) Error() string {
	return "SandboxTimeoutError: " + e.Exception
}

// ClientClosedError is returned when Sandbox operations exceed the allowed time limit.
type ClientClosedError struct {
	Exception string
}

func (e ClientClosedError) Error() string {
	return "ClientClosedError: " + e.Exception
}

// ExecTimeoutError is returned when a container exec exceeds its execution duration limit.
type ExecTimeoutError struct {
	Exception string
}

func (e ExecTimeoutError) Error() string {
	return "ExecTimeoutError: " + e.Exception
}
