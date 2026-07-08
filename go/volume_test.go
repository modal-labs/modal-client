package modal

import (
	"testing"

	"github.com/onsi/gomega"
)

func TestVolumeWithMountOptionsSubPath(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	volume := &Volume{VolumeID: "vo-test"}

	models := "/models"
	subPathVolume := volume.WithMountOptions(&VolumeMountOptionsParams{SubPath: &models})

	proto := volumeToMountProto("/mnt", subPathVolume)
	g.Expect(proto.GetSubPath()).To(gomega.Equal("/models"))
	g.Expect(proto.GetReadOnly()).To(gomega.BeFalse())
	g.Expect(subPathVolume.VolumeID).To(gomega.Equal(volume.VolumeID))
}

func TestVolumeMountOptionsStacking(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	volume := &Volume{VolumeID: "vo-test"}

	trueVal := true
	falseVal := false
	nestedPath := "/nested"
	otherPath := "/other"
	rootPath := "/"

	configured := volume.WithMountOptions(&VolumeMountOptionsParams{
		ReadOnly: &trueVal,
		SubPath:  &nestedPath,
	})

	// Setting only SubPath preserves ReadOnly from the previous call.
	withNewSubPath := configured.WithMountOptions(&VolumeMountOptionsParams{SubPath: &otherPath})
	proto := volumeToMountProto("/mnt", withNewSubPath)
	g.Expect(proto.GetReadOnly()).To(gomega.BeTrue())
	g.Expect(proto.GetSubPath()).To(gomega.Equal("/other"))

	// Setting only ReadOnly preserves SubPath from the previous call.
	withReadOnlyDisabled := configured.WithMountOptions(&VolumeMountOptionsParams{ReadOnly: &falseVal})
	proto = volumeToMountProto("/mnt", withReadOnlyDisabled)
	g.Expect(proto.GetReadOnly()).To(gomega.BeFalse())
	g.Expect(proto.GetSubPath()).To(gomega.Equal("/nested"))

	// SubPath "/" is normalized to "" (mount the whole volume).
	withClearedSubPath := configured.WithMountOptions(&VolumeMountOptionsParams{SubPath: &rootPath})
	proto = volumeToMountProto("/mnt", withClearedSubPath)
	g.Expect(proto.GetReadOnly()).To(gomega.BeTrue())
	g.Expect(proto.GetSubPath()).To(gomega.Equal(""))
}
