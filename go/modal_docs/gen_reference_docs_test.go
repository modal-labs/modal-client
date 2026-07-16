package main

import (
	"os"
	"path/filepath"
	"testing"

	. "github.com/onsi/gomega"
)

// fixturePackage is a miniature `modal` package exercising the doc generator's
// key behaviors: a service interface whose doc comments live on an unexported
// *ServiceImpl, an object type with its own method, *Params options folded into
// their methods, a package-level constructor, and an enum-like defined type.
const fixturePackage = `package modal

import "context"

// Client exposes services for interacting with Modal resources. The accessor
// field (Volumes) intentionally differs from the service type name so the tests
// exercise reading the call path from the struct rather than deriving it.
type Client struct {
	Volumes VolumeService
}

// VolumeService provides Volume operations.
type VolumeService interface {
	FromName(ctx context.Context, name string, params *VolumeFromNameParams) (*Volume, error)
}

type volumeServiceImpl struct{}

// FromName references a Volume by its name.
func (s *volumeServiceImpl) FromName(ctx context.Context, name string, params *VolumeFromNameParams) (*Volume, error) {
	return nil, nil
}

// VolumeFromNameParams are options for FromName.
type VolumeFromNameParams struct {
	Environment     string // Environment to look in.
	CreateIfMissing bool   // Create the Volume if it is missing.
}

// Volume represents a Modal Volume.
type Volume struct {
	VolumeID string
	internal int

	// Filesystem provides filesystem operations for this Volume.
	Filesystem *VolumeFilesystem
}

// VolumeFilesystem operates on a Volume's files.
type VolumeFilesystem struct{ volume *Volume }

// ListFiles lists files in the Volume. See [Volume.Reload] and [StdioBehavior].
func (f *VolumeFilesystem) ListFiles(params *VolumeListFilesParams) ([]string, error) { return nil, nil }

// VolumeListFilesParams are options for ListFiles.
type VolumeListFilesParams struct {
	Recursive bool // Recurse into subdirectories.
}

// Reload reloads the Volume metadata.
func (v *Volume) Reload(params *VolumeReloadParams) error { return nil }

// VolumeReloadParams are options for Reload.
type VolumeReloadParams struct {
	Force bool // Force a reload.
}

// Drop discards the Volume.
func (v *Volume) Drop(params *VolumeDropParams) error { return nil }

// VolumeDropParams are options for Drop.
type VolumeDropParams struct{}

// NewVolume constructs a detached Volume.
func NewVolume() (*Volume, error) { return nil, nil }

// StdioBehavior controls stream behavior.
type StdioBehavior string

const (
	// Pipe pipes the streams.
	Pipe StdioBehavior = "pipe"
	// Ignore drops the streams.
	Ignore StdioBehavior = "ignore"
)

// NotFoundError indicates a missing resource.
type NotFoundError struct {
	Message string
}

func (e NotFoundError) Error() string { return e.Message }

// InternalFailure does not end in "Error" but is still an error type.
type InternalFailure struct{ Code int }

func (e InternalFailure) Error() string { return "internal failure" }
`

func TestGeneratorMergesServiceAndFoldsParams(t *testing.T) {
	g := NewWithT(t)

	srcDir := t.TempDir()
	g.Expect(os.WriteFile(filepath.Join(srcDir, "modal.go"), []byte(fixturePackage), 0o644)).To(Succeed())

	model, err := parsePackage(srcDir)
	g.Expect(err).NotTo(HaveOccurred())

	pages := model.buildPages()
	byLabel := map[string]*page{}
	for _, p := range pages {
		byLabel[p.label] = p
	}

	// Service interface is merged into the object page; neither it nor any
	// *Params type gets a page of its own.
	g.Expect(byLabel).To(HaveKey("Volume"))
	g.Expect(byLabel).NotTo(HaveKey("VolumeService"))
	g.Expect(byLabel).NotTo(HaveKey("VolumeFromNameParams"))
	g.Expect(byLabel).NotTo(HaveKey("VolumeReloadParams"))

	volume := model.render(byLabel["Volume"])

	// Merged service method, with its doc taken from the *ServiceImpl method.
	g.Expect(volume).To(ContainSubstring("## FromName"))
	g.Expect(volume).To(ContainSubstring("FromName references a Volume by its name."))
	// Merged service methods are reached through a Client field rather than the
	// value itself, so they are called out with their access path (a sentence
	// fragment, no trailing period). Methods on the type itself get no caption.
	g.Expect(volume).To(ContainSubstring("_Accessed via `client.Volumes`_"))
	g.Expect(volume).NotTo(ContainSubstring("_Method on"))
	// Folded params with field docs.
	g.Expect(volume).To(ContainSubstring("**Parameters** (`VolumeFromNameParams`)"))
	g.Expect(volume).To(ContainSubstring("`Environment` (`string`): Environment to look in."))
	// Object method and the constructor are on the same page.
	g.Expect(volume).To(ContainSubstring("## Reload"))
	g.Expect(volume).To(ContainSubstring("## NewVolume"))
	g.Expect(volume).To(ContainSubstring("func NewVolume() (*Volume, error)"))
	// An empty params struct is still documented: the type, its doc, and a note.
	g.Expect(volume).To(ContainSubstring("## Drop"))
	g.Expect(volume).To(ContainSubstring("**Parameters** (`VolumeDropParams`)"))
	g.Expect(volume).To(ContainSubstring("VolumeDropParams are options for Drop."))
	g.Expect(volume).To(ContainSubstring("_No configurable options._"))

	// A companion type reached via a field is folded in as a namespace section
	// (with its methods nested deeper), not given a page of its own.
	g.Expect(byLabel).NotTo(HaveKey("VolumeFilesystem"))
	g.Expect(volume).To(ContainSubstring("## Volume.Filesystem"))
	g.Expect(volume).To(ContainSubstring("Filesystem provides filesystem operations for this Volume."))
	g.Expect(volume).To(ContainSubstring("### ListFiles"))
	// godoc symbol references in prose become Markdown code spans; the raw
	// bracket syntax (which the docs site does not render) does not survive.
	g.Expect(volume).To(ContainSubstring("See `Volume.Reload` and `StdioBehavior`."))
	g.Expect(volume).NotTo(ContainSubstring("[Volume.Reload]"))
	// Only exported struct fields appear.
	g.Expect(volume).To(ContainSubstring("VolumeID"))
	g.Expect(volume).NotTo(ContainSubstring("internal"))
}

func TestGeneratorRendersEnumValues(t *testing.T) {
	g := NewWithT(t)

	srcDir := t.TempDir()
	g.Expect(os.WriteFile(filepath.Join(srcDir, "modal.go"), []byte(fixturePackage), 0o644)).To(Succeed())

	model, err := parsePackage(srcDir)
	g.Expect(err).NotTo(HaveOccurred())

	var stdio *page
	for _, p := range model.buildPages() {
		if p.label == "StdioBehavior" {
			stdio = p
		}
	}
	g.Expect(stdio).NotTo(BeNil())

	out := model.render(stdio)
	g.Expect(out).To(ContainSubstring("type StdioBehavior string"))
	g.Expect(out).To(ContainSubstring("The possible values are:"))
	g.Expect(out).To(ContainSubstring("`Pipe` = `\"pipe\"`"))
	g.Expect(out).To(ContainSubstring("Pipe pipes the streams."))
}

func TestGeneratorGroupsErrorTypes(t *testing.T) {
	g := NewWithT(t)

	srcDir := t.TempDir()
	g.Expect(os.WriteFile(filepath.Join(srcDir, "modal.go"), []byte(fixturePackage), 0o644)).To(Succeed())

	model, err := parsePackage(srcDir)
	g.Expect(err).NotTo(HaveOccurred())

	byLabel := map[string]*page{}
	for _, p := range model.buildPages() {
		byLabel[p.label] = p
	}

	// Error types are collected onto a single Errors page, not one page each,
	// detected by the presence of an Error() method (so InternalFailure, which
	// does not end in "Error", is still grouped).
	g.Expect(byLabel).To(HaveKey("Errors"))
	g.Expect(byLabel).NotTo(HaveKey("NotFoundError"))
	g.Expect(byLabel).NotTo(HaveKey("InternalFailure"))

	errs := model.render(byLabel["Errors"])
	g.Expect(errs).To(ContainSubstring("## NotFoundError"))
	g.Expect(errs).To(ContainSubstring("indicates a missing resource"))
	g.Expect(errs).To(ContainSubstring("## InternalFailure"))
}

func TestWriteOutputLayout(t *testing.T) {
	g := NewWithT(t)

	srcDir := t.TempDir()
	g.Expect(os.WriteFile(filepath.Join(srcDir, "modal.go"), []byte(fixturePackage), 0o644)).To(Succeed())

	model, err := parsePackage(srcDir)
	g.Expect(err).NotTo(HaveOccurred())

	outDir := t.TempDir()
	g.Expect(writeOutput(outDir, model, model.buildPages())).To(Succeed())

	// Pages are written flat into the output dir, with the landing page and
	// sidebar index alongside them. The service interface is merged, so it has
	// no page.
	g.Expect(filepath.Join(outDir, "Volume.md")).To(BeAnExistingFile())
	g.Expect(filepath.Join(outDir, "intro.md")).To(BeAnExistingFile())
	g.Expect(filepath.Join(outDir, "sidebar.json")).To(BeAnExistingFile())
	g.Expect(filepath.Join(outDir, "VolumeService.md")).NotTo(BeAnExistingFile())
}

func TestGodocRefsToCode(t *testing.T) {
	g := NewWithT(t)

	cases := map[string]string{
		"see [Sandbox]":             "see `Sandbox`",
		"call [Sandbox.Detach] now": "call `Sandbox.Detach` now",
		"a [*Volume] pointer":       "a `*Volume` pointer",
		"a [time.Duration] value":   "a `time.Duration` value",
		"[A], [b.C], and [D]":       "`A`, `b.C`, and `D`",
		// Multi-word bracketed text (e.g. godoc link definitions) is left alone.
		"[Go blog]: https://go.dev": "[Go blog]: https://go.dev",
		"an [empty] slice []int":    "an `empty` slice []int",
	}
	for in, want := range cases {
		g.Expect(godocRefsToCode(in)).To(Equal(want), "input %q", in)
	}
}
