package modal

import (
	"context"
	"fmt"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// VolumeService provides Volume related operations.
type VolumeService interface {
	FromName(ctx context.Context, name string, params *VolumeFromNameParams) (*Volume, error)
	Ephemeral(ctx context.Context, params *VolumeEphemeralParams) (*Volume, error)
	Delete(ctx context.Context, name string, params *VolumeDeleteParams) error
}

type volumeServiceImpl struct{ client *Client }

// Volume represents a Modal Volume that provides persistent storage.
type Volume struct {
	VolumeID        string
	Name            string
	readOnly        bool
	cancelEphemeral context.CancelFunc
}

// VolumeFromNameParams are options for finding Modal Volumes.
type VolumeFromNameParams struct {
	Environment     string
	CreateIfMissing bool
}

// FromName references a Volume by its name.
func (s *volumeServiceImpl) FromName(ctx context.Context, name string, params *VolumeFromNameParams) (*Volume, error) {
	if params == nil {
		params = &VolumeFromNameParams{}
	}

	creationType := pb.ObjectCreationType_OBJECT_CREATION_TYPE_UNSPECIFIED
	if params.CreateIfMissing {
		creationType = pb.ObjectCreationType_OBJECT_CREATION_TYPE_CREATE_IF_MISSING
	}

	resp, err := s.client.cpClient.VolumeGetOrCreate(ctx, pb.VolumeGetOrCreateRequest_builder{
		DeploymentName:     name,
		EnvironmentName:    environmentName(params.Environment, s.client.profile),
		ObjectCreationType: creationType,
	}.Build())

	if status, ok := status.FromError(err); ok && status.Code() == codes.NotFound {
		return nil, NotFoundError{fmt.Sprintf("Volume '%s' not found", name)}
	}
	if err != nil {
		return nil, err
	}

	s.client.logger.DebugContext(ctx, "Retrieved Volume", "volume_id", resp.GetVolumeId(), "volume_name", name)
	return &Volume{VolumeID: resp.GetVolumeId(), Name: name, readOnly: false, cancelEphemeral: nil}, nil
}

// ReadOnly configures Volume to mount as read-only.
func (v *Volume) ReadOnly() *Volume {
	return &Volume{
		VolumeID:        v.VolumeID,
		Name:            v.Name,
		readOnly:        true,
		cancelEphemeral: v.cancelEphemeral,
	}
}

// IsReadOnly returns true if the Volume is configured to mount as read-only.
func (v *Volume) IsReadOnly() bool {
	return v.readOnly
}

// VolumeEphemeralParams are options for client.Volumes.Ephemeral.
type VolumeEphemeralParams struct {
	Environment string
}

// VolumeDeleteParams are options for client.Volumes.Delete.
type VolumeDeleteParams struct {
	Environment  string
	AllowMissing bool
}

// Ephemeral creates a nameless, temporary Volume, that persists until CloseEphemeral is called, or the process exits.
func (s *volumeServiceImpl) Ephemeral(ctx context.Context, params *VolumeEphemeralParams) (*Volume, error) {
	if params == nil {
		params = &VolumeEphemeralParams{}
	}

	resp, err := s.client.cpClient.VolumeGetOrCreate(ctx, pb.VolumeGetOrCreateRequest_builder{
		ObjectCreationType: pb.ObjectCreationType_OBJECT_CREATION_TYPE_EPHEMERAL,
		EnvironmentName:    environmentName(params.Environment, s.client.profile),
	}.Build())
	if err != nil {
		return nil, err
	}

	s.client.logger.DebugContext(ctx, "Created ephemeral Volume", "volume_id", resp.GetVolumeId())

	ephemeralCtx, cancel := context.WithCancel(context.Background())
	startEphemeralHeartbeat(ephemeralCtx, func() error {
		_, err := s.client.cpClient.VolumeHeartbeat(ephemeralCtx, pb.VolumeHeartbeatRequest_builder{
			VolumeId: resp.GetVolumeId(),
		}.Build())
		return err
	})

	return &Volume{
		VolumeID:        resp.GetVolumeId(),
		readOnly:        false,
		cancelEphemeral: cancel,
	}, nil
}

// CloseEphemeral deletes an ephemeral Volume, only used with VolumeEphemeral.
func (v *Volume) CloseEphemeral() {
	if v.cancelEphemeral != nil {
		v.cancelEphemeral()
	} else {
		// We panic in this case because of invalid usage. In general, methods
		// used with `defer` like CloseEphemeral should not return errors.
		panic(fmt.Sprintf("Volume %s is not ephemeral", v.VolumeID))
	}
}

// Delete deletes a named Volume.
//
// Warning: Deletion is irreversible and will affect any Apps currently using the Volume.
func (s *volumeServiceImpl) Delete(ctx context.Context, name string, params *VolumeDeleteParams) error {
	if params == nil {
		params = &VolumeDeleteParams{}
	}

	volume, err := s.FromName(ctx, name, &VolumeFromNameParams{
		Environment:     params.Environment,
		CreateIfMissing: false,
	})

	if err != nil {
		if _, ok := err.(NotFoundError); ok && params.AllowMissing {
			return nil
		}
		return err
	}

	_, err = s.client.cpClient.VolumeDelete(ctx, pb.VolumeDeleteRequest_builder{
		VolumeId: volume.VolumeID,
	}.Build())

	if err != nil {
		if st, ok := status.FromError(err); ok && st.Code() == codes.NotFound && params.AllowMissing {
			return nil
		}
		return err
	}

	s.client.logger.DebugContext(ctx, "Deleted Volume", "volume_name", name, "volume_id", volume.VolumeID)
	return nil
}
