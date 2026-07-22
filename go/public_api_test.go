package modal

// Tests that public API methods follow the *XxxParams options convention.
// Every public method must end with a *XxxParams pointer as its last argument,
// unless it appears in skipMethods (intentionally exempt).
//
// When adding a new public method:
//   a) End it with a *XxxParams argument — preferred, or
//   b) Add it to skipMethods with a reason if it is intentionally exempt
//      (e.g. it implements a standard interface).
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
	// Simple value accessors.
	"Function.GetWebURL": "returns cached URL string",

	// Returns a copy with straightforward mount options.
	"Volume.WithMountOptions": "creates a copy with mount options, no config options needed",

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

	// Function.Instance does not take in any SDK options itself, rather it uses the existing
	// configuration on the calling struct
	"Function.Instance": "function instantiation does not take any options",

	// args/kwargs are user payload, not SDK options; a params struct would add noise.
	"Function.Remote": "user payload args/kwargs, no SDK options needed",
	"Function.Spawn":  "user payload args/kwargs, no SDK options needed",
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
	// Service interfaces accessed through Client or Sandbox fields.
	{name: "AppService", typ: reflect.TypeOf((*AppService)(nil)).Elem(), isInterface: true},
	{name: "CloudBucketMountService", typ: reflect.TypeOf((*CloudBucketMountService)(nil)).Elem(), isInterface: true},
	{name: "ClsService", typ: reflect.TypeOf((*ClsService)(nil)).Elem(), isInterface: true},
	{name: "FunctionService", typ: reflect.TypeOf((*FunctionService)(nil)).Elem(), isInterface: true},
	{name: "FunctionCallService", typ: reflect.TypeOf((*FunctionCallService)(nil)).Elem(), isInterface: true},
	{name: "ImageService", typ: reflect.TypeOf((*ImageService)(nil)).Elem(), isInterface: true},
	{name: "ProxyService", typ: reflect.TypeOf((*ProxyService)(nil)).Elem(), isInterface: true},
	{name: "QueueService", typ: reflect.TypeOf((*QueueService)(nil)).Elem(), isInterface: true},
	{name: "SandboxService", typ: reflect.TypeOf((*SandboxService)(nil)).Elem(), isInterface: true},
	{name: "SandboxSnapshotService", typ: reflect.TypeOf((*SandboxSnapshotService)(nil)).Elem(), isInterface: true},
	{name: "SidecarService", typ: reflect.TypeOf((*SidecarService)(nil)).Elem(), isInterface: true},
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
	{name: "Volume", typ: reflect.TypeOf((*Volume)(nil)), isInterface: false},
	{name: "Tunnel", typ: reflect.TypeOf((*Tunnel)(nil)), isInterface: false},
	{name: "SandboxFilesystem", typ: reflect.TypeOf((*SandboxFilesystem)(nil)), isInterface: false},
	{name: "SidecarContainer", typ: reflect.TypeOf((*SidecarContainer)(nil)), isInterface: false},
}

// excludedTypes lists exported types that are intentionally absent from
// typeRegistry. These are types with no public methods to check, internal
// helpers, or value/data types.
//
// Types whose names end in "Params" or "Error" are automatically excluded and
// do not need to appear here.
var excludedTypes = map[string]string{
	"App":                             "data type, no public methods",
	"Client":                          "main client; Close and Version are intentionally parameter-free",
	"CloudBucketMount":                "data type, no public methods",
	"FunctionStats":                   "data type",
	"Profile":                         "data type, no public methods",
	"Probe":                           "configuration value type, no public methods",
	"Proxy":                           "data type, no public methods",
	"Retries":                         "configuration value type, no public methods",
	"SandboxCreateConnectCredentials": "data type",
	"SandboxSnapshot":                 "data type, no public methods",
	"Secret":                          "data type, no public methods",
	"StdioBehavior":                   "string enum type",
	"InternalFailure":                 "error type (name does not end in Error)",
	"Allowlist":                       "configuration value type, no public methods",
	"FileType":                        "string enum, no public methods",
	"FileInfo":                        "data type, no public methods",
	"FileWatchEventType":              "string enum, no public methods",
	"FileWatchEvent":                  "data type, no public methods",
	"VolumeMountOptionsParams":        "configuration value type",
}

// isParamsType reports whether t is a pointer to a struct whose name ends with "Params".
func isParamsType(t reflect.Type) bool {
	return t.Kind() == reflect.Ptr &&
		t.Elem().Kind() == reflect.Struct &&
		strings.HasSuffix(t.Elem().Name(), "Params")
}

// checkMethodsHaveParams verifies that every public method on typ ends with a
// *XxxParams argument. Methods in skipMethods are permanently exempt. Any other
// non-compliant method fails the test.
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

		if compliant {
			continue
		}

		lastParamStr := "none"
		if numParams > 0 {
			lastParamStr = mt.In(mt.NumIn() - 1).String()
		}
		t.Errorf(
			"public method %s does not end with *XxxParams (params: %d, last: %s).\n"+
				"Add a *XxxParams argument, or add to skipMethods if intentionally exempt.",
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

// TestSkipMethodKeysAreValid ensures every key in skipMethods refers to a type
// present in typeRegistry and a method that actually exists on that type. This
// catches stale entries left behind after a type or method is renamed or removed.
func TestSkipMethodKeysAreValid(t *testing.T) {
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
