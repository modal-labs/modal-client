package modal

// Tests that public API methods follow the *XxxParams options convention.
// Every public method must end with a *XxxParams pointer as its last argument,
// unless it appears in skipMethods (intentionally exempt) or knownParamsViolations
// (pre-existing non-compliance that should be fixed over time).
//
// When adding a new public method:
//   a) End it with a *XxxParams argument — preferred, or
//   b) Add it to skipMethods with a reason if it is intentionally exempt
//      (e.g. it implements a standard interface), or
//   c) Do NOT add to knownParamsViolations — that set must never grow.
//
// When adding a new exported type:
//   a) Add it to typeRegistry to have its methods checked, or
//   b) Add it to excludedTypes if it has no public methods to check.
//   Types whose names end in "Params" or "Error" are automatically excluded.

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// skipMethods lists public methods that are intentionally exempt from the
// *XxxParams rule. These implement standard interfaces or are simple accessors
// that would gain no benefit from an options struct.
var skipMethods = map[string]string{
	// SandboxFile implements io.Reader, io.Writer, and io.Closer.
	"SandboxFile.Read":  "implements io.Reader",
	"SandboxFile.Write": "implements io.Writer",
	"SandboxFile.Flush": "file flush signal, no config needed",
	"SandboxFile.Close": "implements io.Closer",

	// Simple boolean predicates and value accessors.
	"Volume.IsReadOnly":  "boolean predicate accessor",
	"Function.GetWebURL": "returns cached URL string",

	// Returns a modified copy with a single semantically obvious change.
	"Volume.ReadOnly": "creates a read-only copy, no config options needed",

	// Lifecycle / cleanup signals; configuration-free by design.
	"Volume.CloseEphemeral": "lifecycle signal, no config needed",
	"Queue.CloseEphemeral":  "lifecycle signal, no config needed",
	"Sandbox.Detach":        "disconnects from sandbox, no config needed",

	// Tunnel accessors: return structured data derived from the Tunnel's fields.
	"Tunnel.URL":       "returns computed URL string",
	"Tunnel.TLSSocket": "returns (host, port) pair",
	"Tunnel.TCPSocket": "returns (host, port, error) triple",

	// ClsInstance.Method is a local name lookup
	"ClsInstance.Method": "local method lookup by name without network call, no config needed",

	// Cls.Instance takes user-provided constructor parameters, not SDK options.
	"Cls.Instance": "parameters map is user data, not SDK options",
}

// knownParamsViolations lists public methods that do not yet follow the
// *XxxParams convention. Do NOT add new entries — fix violations by adding a
// *XxxParams argument instead.
var knownParamsViolations = map[string]string{
	// FromID methods only take an ID string.
	"FunctionCallService.FromID": "only takes an ID string",
	"ImageService.FromID":        "only takes an ID string",
	"SandboxService.FromID":      "only takes an ID string",

	// Image registry constructors use a *Secret directly instead of *Params.
	"ImageService.FromAwsEcr":              "takes *Secret directly instead of *Params",
	"ImageService.FromGcpArtifactRegistry": "takes *Secret directly instead of *Params",

	// Function invocation methods take serialized user data, not SDK options.
	"Function.Remote": "args/kwargs are function inputs, not SDK options",
	"Function.Spawn":  "args/kwargs are function inputs, not SDK options",

	// Context-only methods that should gain a *Params argument.
	"Function.GetCurrentStats": "context-only, should add *FunctionGetCurrentStatsParams",
	"ContainerProcess.Wait":    "context-only, should add *ContainerProcessWaitParams",

	// Sandbox methods with plain positional args or durations instead of *Params.
	"Sandbox.Open":               "takes path and mode strings",
	"Sandbox.Wait":               "context-only, should add *SandboxWaitParams",
	"Sandbox.WaitUntilReady":     "takes timeout duration directly",
	"Sandbox.Tunnels":            "takes timeout duration directly",
	"Sandbox.MountImage":         "takes path and image as positional args",
	"Sandbox.UnmountImage":       "takes path string",
	"Sandbox.SnapshotDirectory":  "takes path string",
	"Sandbox.SnapshotFilesystem": "takes timeout duration directly",
	"Sandbox.Poll":               "context-only, should add *SandboxPollParams",
	"Sandbox.SetTags":            "takes tags map as primary input",
	"Sandbox.GetTags":            "context-only, should add *SandboxGetTagsParams",

	// Image.Build takes an App as a required dependency, not an options struct.
	"Image.Build": "app is a required dependency, not options",
}

// typeEntry pairs an exported type name with its reflect.Type and whether it is
// an interface (true) or a concrete pointer type (false).
type typeEntry struct {
	name        string
	typ         reflect.Type
	isInterface bool
}

// typeRegistry is the authoritative list of exported types whose public methods
// are checked by TestPublicMethodsHaveParamsArg.
//
// When adding a new exported type that has public methods, add it here.
// If the type has no public methods to check, add it to excludedTypes instead.
var typeRegistry = []typeEntry{
	// Service interfaces accessed through Client fields.
	{name: "AppService", typ: reflect.TypeOf((*AppService)(nil)).Elem(), isInterface: true},
	{name: "CloudBucketMountService", typ: reflect.TypeOf((*CloudBucketMountService)(nil)).Elem(), isInterface: true},
	{name: "ClsService", typ: reflect.TypeOf((*ClsService)(nil)).Elem(), isInterface: true},
	{name: "FunctionService", typ: reflect.TypeOf((*FunctionService)(nil)).Elem(), isInterface: true},
	{name: "FunctionCallService", typ: reflect.TypeOf((*FunctionCallService)(nil)).Elem(), isInterface: true},
	{name: "ImageService", typ: reflect.TypeOf((*ImageService)(nil)).Elem(), isInterface: true},
	{name: "ProxyService", typ: reflect.TypeOf((*ProxyService)(nil)).Elem(), isInterface: true},
	{name: "QueueService", typ: reflect.TypeOf((*QueueService)(nil)).Elem(), isInterface: true},
	{name: "SandboxService", typ: reflect.TypeOf((*SandboxService)(nil)).Elem(), isInterface: true},
	{name: "SecretService", typ: reflect.TypeOf((*SecretService)(nil)).Elem(), isInterface: true},
	{name: "VolumeService", typ: reflect.TypeOf((*VolumeService)(nil)).Elem(), isInterface: true},

	// Object types returned by service methods (pointer receivers).
	{name: "Cls", typ: reflect.TypeOf((*Cls)(nil)), isInterface: false},
	{name: "ClsInstance", typ: reflect.TypeOf((*ClsInstance)(nil)), isInterface: false},
	{name: "ContainerProcess", typ: reflect.TypeOf((*ContainerProcess)(nil)), isInterface: false},
	{name: "Function", typ: reflect.TypeOf((*Function)(nil)), isInterface: false},
	{name: "FunctionCall", typ: reflect.TypeOf((*FunctionCall)(nil)), isInterface: false},
	{name: "Image", typ: reflect.TypeOf((*Image)(nil)), isInterface: false},
	{name: "Queue", typ: reflect.TypeOf((*Queue)(nil)), isInterface: false},
	{name: "Sandbox", typ: reflect.TypeOf((*Sandbox)(nil)), isInterface: false},
	{name: "SandboxFile", typ: reflect.TypeOf((*SandboxFile)(nil)), isInterface: false},
	{name: "Volume", typ: reflect.TypeOf((*Volume)(nil)), isInterface: false},
	{name: "Tunnel", typ: reflect.TypeOf((*Tunnel)(nil)), isInterface: false},
}

// excludedTypes lists exported types that are intentionally absent from
// typeRegistry. These are types with no public methods to check, internal
// helpers, or value/data types.
//
// Types whose names end in "Params" or "Error" are automatically excluded and
// do not need to appear here.
var excludedTypes = map[string]string{
	"App":                             "data type, no public methods",
	"AuthTokenManager":                "internal token management, not SDK API surface",
	"Client":                          "main client; Close and Version are intentionally parameter-free",
	"CloudBucketMount":                "data type, no public methods",
	"FunctionStats":                   "data type",
	"Profile":                         "data type, no public methods",
	"Probe":                           "configuration value type, no public methods",
	"Proxy":                           "data type, no public methods",
	"Retries":                         "configuration value type, no public methods",
	"SandboxCreateConnectCredentials": "data type",
	"Secret":                          "data type, no public methods",
	"StdioBehavior":                   "string enum type",
	"TokenAndExpiry":                  "internal data type",
	"InternalFailure":                 "error type (name does not end in Error)",
}

// isParamsType reports whether t is a pointer to a struct whose name ends with "Params".
func isParamsType(t reflect.Type) bool {
	return t.Kind() == reflect.Ptr &&
		t.Elem().Kind() == reflect.Struct &&
		strings.HasSuffix(t.Elem().Name(), "Params")
}

// checkMethodsHaveParams verifies that every public method on typ ends with a
// *XxxParams argument. Methods in skipMethods are permanently exempt. Methods
// in knownParamsViolations are pre-existing issues. Any other non-compliant
// method fails the test.
//
// isInterface must be true for interface types (no receiver in the method signature)
// and false for concrete pointer types (receiver is In(0)).
func checkMethodsHaveParams(t *testing.T, typeName string, typ reflect.Type, isInterface bool) {
	t.Helper()
	for i := range typ.NumMethod() {
		m := typ.Method(i)
		key := typeName + "." + m.Name

		if _, ok := skipMethods[key]; ok {
			continue
		}

		mt := m.Type
		offset := 0
		if !isInterface {
			offset = 1 // skip the receiver
		}
		numParams := mt.NumIn() - offset

		compliant := numParams >= 1 && isParamsType(mt.In(mt.NumIn()-1))

		if _, ok := knownParamsViolations[key]; ok {
			if compliant {
				t.Errorf("knownParamsViolations entry %q is stale: method is now compliant. Remove it.", key)
			}
			continue
		}

		if compliant {
			continue
		}

		lastParamStr := "none"
		if numParams > 0 {
			lastParamStr = mt.In(mt.NumIn() - 1).String()
		}
		t.Errorf(
			"public method %s does not end with *XxxParams (params: %d, last: %s).\n"+
				"Add a *XxxParams argument, add to skipMethods if intentionally exempt,\n"+
				"or to knownParamsViolations if it cannot be changed right now.",
			key, numParams, lastParamStr,
		)
	}
}

func TestPublicMethodsHaveParamsArg(t *testing.T) {
	for _, e := range typeRegistry {
		checkMethodsHaveParams(t, e.name, e.typ, e.isInterface)
	}
}

// parseExportedTypeNames parses all non-test .go files in the package directory
// and returns the names of every exported type declaration.
// sourceDir returns the directory containing the package's non-test .go source
// files. Under normal "go test" the working directory is already the package
// directory. Under Bazel, source files listed in data are placed under
// $TEST_SRCDIR/<workspace>/<package-path>, so we reconstruct that path.
func sourceDir() string {
	if testSrcdir := os.Getenv("TEST_SRCDIR"); testSrcdir != "" {
		workspace := os.Getenv("TEST_WORKSPACE")
		if workspace == "" {
			workspace = "_main"
		}
		return filepath.Join(testSrcdir, workspace, "client", "go")
	}
	return "."
}

func parseExportedTypeNames(t *testing.T) map[string]bool {
	t.Helper()
	dir := sourceDir()
	fset := token.NewFileSet()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("parseExportedTypeNames: read dir %q: %v", dir, err)
	}
	names := map[string]bool{}
	filesFound := 0
	for _, entry := range entries {
		name := entry.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		filesFound++
		f, err := parser.ParseFile(fset, filepath.Join(dir, name), nil, 0)
		if err != nil {
			t.Fatalf("parseExportedTypeNames: parse %s: %v", name, err)
		}
		for _, decl := range f.Decls {
			gd, ok := decl.(*ast.GenDecl)
			if !ok || gd.Tok != token.TYPE {
				continue
			}
			for _, spec := range gd.Specs {
				ts, ok := spec.(*ast.TypeSpec)
				if ok && ts.Name.IsExported() {
					names[ts.Name.Name] = true
				}
			}
		}
	}
	if filesFound == 0 {
		t.Fatalf(
			"parseExportedTypeNames: no .go source files found in %q — "+
				"under Bazel, the go_test data attribute must include the library sources",
			dir,
		)
	}
	return names
}

// TestSkipAndViolationKeysAreValid ensures every key in skipMethods and
// knownParamsViolations refers to a type present in typeRegistry and a method
// that actually exists on that type. This catches stale entries left behind
// after a type or method is renamed or removed.
func TestSkipAndViolationKeysAreValid(t *testing.T) {
	byName := map[string]typeEntry{}
	for _, e := range typeRegistry {
		byName[e.name] = e
	}

	checkKey := func(key, source string) {
		t.Helper()
		typeName, methodName, ok2 := strings.Cut(key, ".")
		if !ok2 {
			t.Errorf("%s key %q has no dot separator", source, key)
			return
		}
		e, ok := byName[typeName]
		if !ok {
			t.Errorf("%s key %q: type %q not found in typeRegistry", source, key, typeName)
			return
		}
		for i := range e.typ.NumMethod() {
			if e.typ.Method(i).Name == methodName {
				return
			}
		}
		t.Errorf("%s key %q: method %q not found on type %q", source, key, methodName, typeName)
	}

	for key := range skipMethods {
		checkKey(key, "skipMethods")
	}
	for key := range knownParamsViolations {
		checkKey(key, "knownParamsViolations")
	}
}

// TestAllExportedTypesRegistered ensures every exported type in the package is
// accounted for: either in typeRegistry (its methods are checked) or in
// excludedTypes (intentionally not checked). Types whose names end in "Params"
// or "Error" are automatically excluded.
//
// This test fails when a new exported type is added without updating one of
// these maps, preventing new types from silently bypassing the params check.
func TestAllExportedTypesRegistered(t *testing.T) {
	all := parseExportedTypeNames(t)

	registered := map[string]bool{}
	for _, e := range typeRegistry {
		registered[e.name] = true
	}

	for name := range all {
		if registered[name] {
			continue
		}
		if _, ok := excludedTypes[name]; ok {
			continue
		}
		if strings.HasSuffix(name, "Params") || strings.HasSuffix(name, "Error") {
			continue
		}
		t.Errorf(
			"exported type %s is unaccounted for.\n"+
				"Add it to typeRegistry if its methods should be checked for *XxxParams,\n"+
				"or add it to excludedTypes if it has no methods to check.",
			name,
		)
	}

	for name := range excludedTypes {
		if !all[name] {
			t.Errorf(
				"excludedTypes entry %q does not refer to any exported type in the package.\n"+
					"Remove it if the type no longer exists.",
				name,
			)
		}
	}
}
