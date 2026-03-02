package modal

import (
	"context"
	"io"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
)

// SandboxFile represents an open file in the Sandbox filesystem.
// It implements io.Reader, io.Writer, io.Seeker, and io.Closer interfaces.
type SandboxFile struct {
	fileDescriptor string
	taskID         string
	cpClient       pb.ModalClientClient
}

// Read reads up to len(p) bytes from the file into p.
// It returns the number of bytes read and any error encountered.
func (f *SandboxFile) Read(p []byte) (int, error) {
	nBytes := uint32(len(p))
	totalRead, _, err := runFilesystemExec(context.Background(), f.cpClient, pb.ContainerFilesystemExecRequest_builder{
		FileReadRequest: pb.ContainerFileReadRequest_builder{
			FileDescriptor: f.fileDescriptor,
			N:              &nBytes,
		}.Build(),
		TaskId: f.taskID,
	}.Build(), p)
	if err != nil {
		return 0, err
	}
	if totalRead < int(nBytes) {
		return totalRead, io.EOF
	}
	return totalRead, nil
}

// Write writes len(p) bytes from p to the file.
// It returns the number of bytes written and any error encountered.
func (f *SandboxFile) Write(p []byte) (n int, err error) {
	_, _, err = runFilesystemExec(context.Background(), f.cpClient, pb.ContainerFilesystemExecRequest_builder{
		FileWriteRequest: pb.ContainerFileWriteRequest_builder{
			FileDescriptor: f.fileDescriptor,
			Data:           p,
		}.Build(),
		TaskId: f.taskID,
	}.Build(), nil)
	if err != nil {
		return 0, err
	}
	return len(p), nil
}

// Flush flushes any buffered data to the file.
func (f *SandboxFile) Flush() error {
	_, _, err := runFilesystemExec(context.Background(), f.cpClient, pb.ContainerFilesystemExecRequest_builder{
		FileFlushRequest: pb.ContainerFileFlushRequest_builder{
			FileDescriptor: f.fileDescriptor,
		}.Build(),
		TaskId: f.taskID,
	}.Build(), nil)
	if err != nil {
		return err
	}
	return nil
}

// Close closes the file, rendering it unusable for I/O.
func (f *SandboxFile) Close() error {
	_, _, err := runFilesystemExec(context.Background(), f.cpClient, pb.ContainerFilesystemExecRequest_builder{
		FileCloseRequest: pb.ContainerFileCloseRequest_builder{
			FileDescriptor: f.fileDescriptor,
		}.Build(),
		TaskId: f.taskID,
	}.Build(), nil)
	if err != nil {
		return err
	}
	return nil
}

func runFilesystemExec(ctx context.Context, cpClient pb.ModalClientClient, req *pb.ContainerFilesystemExecRequest, p []byte) (int, *pb.ContainerFilesystemExecResponse, error) {
	resp, err := cpClient.ContainerFilesystemExec(ctx, req)
	if err != nil {
		return 0, nil, err
	}
	retries := 10
	totalRead := 0

	streamCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	for {
		outputIterator, err := cpClient.ContainerFilesystemExecGetOutput(streamCtx, pb.ContainerFilesystemExecGetOutputRequest_builder{
			ExecId:  resp.GetExecId(),
			Timeout: 55,
		}.Build())
		if err != nil {
			if isRetryableGrpc(err) && retries > 0 {
				retries--
				continue
			}
			return 0, nil, err
		}

		for {
			batch, err := outputIterator.Recv()
			if err == io.EOF {
				break
			}
			if err != nil {
				if isRetryableGrpc(err) && retries > 0 {
					retries--
					break
				}
				return 0, nil, err
			}
			if batch.GetError() != nil {
				return 0, nil, SandboxFilesystemError{batch.GetError().GetErrorMessage()}
			}

			for _, chunk := range batch.GetOutput() {
				copyLen := copy(p[totalRead:], chunk)
				totalRead += copyLen
			}

			if batch.GetEof() {
				return totalRead, resp, nil
			}
		}
	}
}
