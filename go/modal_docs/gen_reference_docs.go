// Command modal_docs generates Markdown reference documentation for the public
// Go SDK in client/go (package modal).
//
// It mirrors the Python reference pipeline (client/py/modal_docs): one Markdown
// page per primary type, written so the docs site can render them under
// /docs/sdk/go/latest/<Type>.
//
// Four Go-specific conventions shape the output:
//
//   - Each resource exposes both an object type (e.g. Volume) and a service
//     interface (e.g. VolumeService, reached via client.Volumes). The service
//     interface's methods are merged onto the object-type page, so a reader sees
//     FromName/Ephemeral/Delete next to Volume's own methods. The doc comments
//     for service methods live on the unexported *ServiceImpl methods, which we
//     match to the interface methods by name.
//   - Every *XxxParams options type is folded into the page section of the
//     method whose final argument is *XxxParams (the convention enforced by
//     public_api_test.go), rather than getting a page of its own.
//   - Types that implement error (declare an Error() method) are collected onto
//     a single Errors page rather than getting a page each.
//   - A "companion" type reached through a field on a primary type (e.g.
//     Sandbox.Filesystem -> *SandboxFilesystem) is folded into the primary's
//     page as a namespace section rather than getting a page of its own — the Go
//     analog of Python's mdmd:namespace folding.
//
// Usage:
//
//	go run ./modal_docs <output_dir> [<source_dir>]
//
// <source_dir> defaults to ".", which under `go run ./modal_docs` (cwd
// client/go) is the package directory. Pages are written flat to
// <output_dir>/<Type>.md, a landing page to <output_dir>/intro.md, and a
// sidebar index to <output_dir>/sidebar.json.
package main

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/printer"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"unicode"
)

// hiddenTypes are exported types that are part of the package surface but are
// not user-facing API and should not be documented.
var hiddenTypes = map[string]bool{
	"AuthTokenManager": true,
	"TokenAndExpiry":   true,
}

// hiddenFuncs are exported package-level functions that should not be
// documented (typically constructors of hidden types).
var hiddenFuncs = map[string]bool{
	"NewAuthTokenManager": true,
	"ValidateExecArgs":    true,
}

func main() {
	args := os.Args[1:]
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "usage: modal_docs <output_dir> [<source_dir>]")
		os.Exit(2)
	}
	outDir := args[0]
	srcDir := "."
	if len(args) >= 2 {
		srcDir = args[1]
	}

	model, err := parsePackage(srcDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	pages := model.buildPages()
	if err := writeOutput(outDir, model, pages); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Wrote %d Go reference pages to %s\n", len(pages), outDir)
}

// typeInfo holds a parsed type declaration and any constants of that type.
type typeInfo struct {
	name   string
	doc    string
	spec   *ast.TypeSpec
	consts []constInfo
}

// constInfo is a single constant value belonging to a (usually enum-like) type.
type constInfo struct {
	name  string
	value string
	doc   string
}

// pkgModel is the parsed view of the package needed for doc generation.
type pkgModel struct {
	fset *token.FileSet
	// types is every type declaration keyed by name (exported and unexported).
	types map[string]*typeInfo
	// methods maps a receiver type name (without leading *) to its methods.
	methods map[string]map[string]*ast.FuncDecl
	// funcs is every package-level (non-method) function keyed by name.
	funcs map[string]*ast.FuncDecl
	// serviceAccessors maps a service interface type name to the exported field
	// on the Client struct that exposes it (e.g. "VolumeService" -> "Volumes"),
	// so docs can name the real call path (client.Volumes.FromName). The mapping
	// is not mechanical (VolumeService -> Volumes, but ClsService -> Cls), so it
	// is read from the struct rather than derived from the type name.
	serviceAccessors map[string]string
}

// clientTypeName is the exported struct whose fields expose the service
// interfaces (the SDK entrypoint; see [NewClient]).
const clientTypeName = "Client"

func parsePackage(srcDir string) (*pkgModel, error) {
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, srcDir, func(fi os.FileInfo) bool {
		return !strings.HasSuffix(fi.Name(), "_test.go")
	}, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("parsing %q: %w", srcDir, err)
	}

	pkg, ok := pkgs["modal"]
	if !ok {
		return nil, fmt.Errorf("package %q not found in %q", "modal", srcDir)
	}

	m := &pkgModel{
		fset:             fset,
		types:            map[string]*typeInfo{},
		methods:          map[string]map[string]*ast.FuncDecl{},
		funcs:            map[string]*ast.FuncDecl{},
		serviceAccessors: map[string]string{},
	}

	for _, file := range pkg.Files {
		for _, decl := range file.Decls {
			switch d := decl.(type) {
			case *ast.FuncDecl:
				if d.Recv != nil {
					recv := recvTypeName(d.Recv)
					if recv == "" {
						continue
					}
					if m.methods[recv] == nil {
						m.methods[recv] = map[string]*ast.FuncDecl{}
					}
					m.methods[recv][d.Name.Name] = d
				} else {
					m.funcs[d.Name.Name] = d
				}
			case *ast.GenDecl:
				switch d.Tok {
				case token.TYPE:
					for _, spec := range d.Specs {
						ts, ok := spec.(*ast.TypeSpec)
						if !ok {
							continue
						}
						doc := ""
						if ts.Doc != nil {
							doc = ts.Doc.Text()
						} else if d.Doc != nil {
							doc = d.Doc.Text()
						}
						m.types[ts.Name.Name] = &typeInfo{name: ts.Name.Name, doc: doc, spec: ts}
					}
				case token.CONST:
					m.collectConsts(d)
				}
			}
		}
	}

	m.buildServiceAccessors()

	return m, nil
}

// buildServiceAccessors records, for each service interface exposed as an
// exported field on the Client struct, the field name that reaches it (e.g.
// client.Volumes -> VolumeService). Doc rendering uses this to show the real
// call path for merged service methods.
func (m *pkgModel) buildServiceAccessors() {
	ti := m.types[clientTypeName]
	if ti == nil {
		return
	}
	st, ok := ti.spec.Type.(*ast.StructType)
	if !ok {
		return
	}
	for _, field := range st.Fields.List {
		if len(field.Names) != 1 || !isExported(field.Names[0].Name) {
			continue
		}
		svcType := baseTypeName(field.Type)
		if svcType == "" {
			continue
		}
		if fieldTI := m.types[svcType]; fieldTI != nil && isServiceInterface(fieldTI) {
			m.serviceAccessors[svcType] = field.Names[0].Name
		}
	}
}

// collectConsts records constants grouped by their declared type, so enum-like
// defined types can list their possible values. Within a single const block the
// type carries forward across specs that omit it (iota continuation).
func (m *pkgModel) collectConsts(d *ast.GenDecl) {
	lastType := ""
	for _, spec := range d.Specs {
		vs, ok := spec.(*ast.ValueSpec)
		if !ok {
			continue
		}
		typeName := lastType
		if id, ok := vs.Type.(*ast.Ident); ok {
			typeName = id.Name
			lastType = id.Name
		}
		if typeName == "" {
			continue
		}
		ti := m.types[typeName]
		if ti == nil {
			continue
		}
		for i, name := range vs.Names {
			if !isExported(name.Name) {
				continue
			}
			value := ""
			if i < len(vs.Values) {
				value = m.printNode(vs.Values[i])
			}
			doc := ""
			if vs.Doc != nil {
				doc = vs.Doc.Text()
			} else if vs.Comment != nil {
				doc = vs.Comment.Text()
			}
			ti.consts = append(ti.consts, constInfo{name: name.Name, value: value, doc: oneLine(doc)})
		}
	}
}

// page is a single output Markdown page for one public type (or, rarely, a
// free function).
type page struct {
	label string // page title and filename stem, e.g. "Volume"
	ti    *typeInfo

	// serviceIface, when set, is a service interface whose methods are merged
	// onto this page (e.g. VolumeService merged onto the Volume page).
	serviceIface *typeInfo
	// serviceOwnPage marks a service interface that has no matching object type
	// and is documented on its own page (e.g. SidecarService).
	serviceOwnPage bool

	constructors []*ast.FuncDecl // package-level constructors returning this type
	freeFunc     *ast.FuncDecl   // set for a standalone-function page

	// groupedTypes, when non-empty, makes this an aggregate page that documents
	// several small related types as sections (used for the Errors page).
	groupedTypes []*typeInfo

	// companions are dependent types reached through a field on this type (e.g.
	// Sandbox.Filesystem), documented inline as namespace sections.
	companions []companion
}

// companion is a dependent type folded into a primary type's page, reached via
// an exported field on that primary (e.g. Sandbox.Filesystem -> SandboxFilesystem).
type companion struct {
	fieldName string // exported field on the primary, e.g. "Filesystem"
	fieldDoc  string // doc comment on that field
	ti        *typeInfo
	isService bool // the companion type is a service interface
}

// errorsPageLabel is the title/URL stem of the aggregate page that collects all
// error types, which are too numerous and small to warrant a page each.
const errorsPageLabel = "Errors"

func (p *page) category() string {
	if p.freeFunc != nil {
		return "function"
	}
	return "type"
}

// buildPages classifies every exported declaration into the set of pages.
func (m *pkgModel) buildPages() []*page {
	pages := map[string]*page{}
	var errorTypes []*typeInfo

	// Object/value/enum types get their own page. Params types are folded into
	// methods; service interfaces are handled separately below; error types are
	// collected onto a single aggregate page.
	for name, ti := range m.types {
		if !isExported(name) || hiddenTypes[name] {
			continue
		}
		if isParamsType(name) || isServiceInterface(ti) {
			continue
		}
		if m.isErrorType(name) {
			errorTypes = append(errorTypes, ti)
			continue
		}
		pages[name] = &page{label: name, ti: ti}
	}

	// Merge each service interface into its matching object-type page, or give
	// it a page of its own when there is no match.
	for name, ti := range m.types {
		if !isServiceInterface(ti) || hiddenTypes[name] {
			continue
		}
		target := strings.TrimSuffix(name, "Service")
		if p, ok := pages[target]; ok {
			p.serviceIface = ti
		} else {
			pages[name] = &page{label: name, ti: ti, serviceIface: ti, serviceOwnPage: true}
		}
	}

	m.foldCompanions(pages)

	// Attach package-level constructors to the page of the type they return;
	// any remaining free function becomes its own page.
	for fname, fd := range m.funcs {
		if !isExported(fname) || hiddenFuncs[fname] {
			continue
		}
		ret := firstResultBaseType(fd)
		if p, ok := pages[ret]; ok && p.ti != nil {
			p.constructors = append(p.constructors, fd)
		} else {
			pages[fname] = &page{label: fname, freeFunc: fd}
		}
	}

	if len(errorTypes) > 0 {
		sort.Slice(errorTypes, func(i, j int) bool { return errorTypes[i].name < errorTypes[j].name })
		pages[errorsPageLabel] = &page{label: errorsPageLabel, groupedTypes: errorTypes}
	}

	out := make([]*page, 0, len(pages))
	for _, p := range pages {
		out = append(out, p)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].label < out[j].label })
	for _, p := range out {
		sort.Slice(p.constructors, func(i, j int) bool {
			return p.constructors[i].Name.Name < p.constructors[j].Name.Name
		})
		sort.Slice(p.companions, func(i, j int) bool {
			return p.companions[i].fieldName < p.companions[j].fieldName
		})
	}
	return out
}

// foldCompanions detects dependent types reached through a field on a primary
// type (e.g. Sandbox.Filesystem -> *SandboxFilesystem) and folds them into the
// primary's page as namespace sections, removing their standalone page. A type
// qualifies when it is referenced via an exported field of a page type and is
// either a service interface or has methods of its own (a concrete operations
// namespace) — plain data fields are left as their own pages. A companion
// reached from several primaries (e.g. both Sandbox and SidecarContainer expose
// .Filesystem) is documented on each so every page is self-contained.
func (m *pkgModel) foldCompanions(pages map[string]*page) {
	type addition struct {
		owner string
		comp  companion
	}
	var additions []addition
	deletions := map[string]bool{}

	for name, p := range pages {
		if p.ti == nil {
			continue
		}
		st, ok := p.ti.spec.Type.(*ast.StructType)
		if !ok {
			continue
		}
		for _, field := range st.Fields.List {
			if len(field.Names) != 1 || !isExported(field.Names[0].Name) {
				continue
			}
			dep := baseTypeName(field.Type)
			if dep == "" || dep == name {
				continue
			}
			if _, ok := pages[dep]; !ok {
				continue // not a standalone page (external type, merged, grouped)
			}
			ti := m.types[dep]
			isSvc := isServiceInterface(ti)
			if !isSvc && !m.hasExportedMethods(dep) {
				continue // plain data type, not an operations namespace
			}
			additions = append(additions, addition{owner: name, comp: companion{
				fieldName: field.Names[0].Name,
				fieldDoc:  fieldDoc(field),
				ti:        ti,
				isService: isSvc,
			}})
			deletions[dep] = true
		}
	}

	for dep := range deletions {
		delete(pages, dep)
	}
	for _, a := range additions {
		// Skip if the owner was itself folded away into another type.
		if owner, ok := pages[a.owner]; ok {
			owner.companions = append(owner.companions, a.comp)
		}
	}
}

// render produces the full Markdown for a page.
func (m *pkgModel) render(p *page) string {
	var b strings.Builder

	fmt.Fprintf(&b, "# %s\n\n", p.label)

	if len(p.groupedTypes) > 0 {
		for _, ti := range p.groupedTypes {
			fmt.Fprintf(&b, "## %s\n\n", ti.name)
			writeDoc(&b, ti.doc)
			if decl := m.typeDecl(ti); decl != "" {
				b.WriteString(decl)
				b.WriteString("\n")
			}
		}
		return b.String()
	}

	if p.freeFunc != nil {
		m.renderFunc(&b, p.freeFunc, 2)
		return b.String()
	}

	writeDoc(&b, p.ti.doc)

	if decl := m.renderTypeDecl(p); decl != "" {
		b.WriteString(decl)
		b.WriteString("\n")
	}

	// Constructors first (how you obtain the value), then service ("factory")
	// methods, then methods on the value itself.
	for _, fd := range p.constructors {
		m.renderFunc(&b, fd, 2)
	}
	if p.serviceIface != nil {
		m.renderServiceMethods(&b, p.serviceIface, 2, m.serviceAccessCaption(p.serviceIface.name))
	}
	m.renderObjectMethods(&b, p.label, 2)

	// Dependent "companion" types reached through a field on this type (e.g.
	// sandbox.Filesystem) are documented inline as a namespace, one heading
	// level below this type's own methods.
	for _, c := range p.companions {
		m.renderCompanion(&b, p.label, c)
	}

	return b.String()
}

// renderCompanion documents a companion type as a namespace section introduced
// by its accessor path (e.g. "## Sandbox.Filesystem"), with the companion's
// methods nested one level deeper.
func (m *pkgModel) renderCompanion(b *strings.Builder, primary string, c companion) {
	fmt.Fprintf(b, "## %s.%s\n\n", primary, c.fieldName)
	doc := strings.TrimSpace(c.fieldDoc)
	if doc == "" {
		doc = c.ti.doc
	}
	writeDoc(b, doc)
	if c.isService {
		m.renderServiceMethods(b, c.ti, 3, "")
	} else {
		m.renderObjectMethods(b, c.ti.name, 3)
	}
}

// renderTypeDecl renders the type's declaration block: exported struct fields,
// or a defined/enum type with its constants. Service-interface method sets are
// rendered as method sections instead, so their decl block is omitted.
func (m *pkgModel) renderTypeDecl(p *page) string {
	if p.serviceOwnPage {
		return ""
	}
	return m.typeDecl(p.ti)
}

// typeDecl renders a single type's declaration block: exported struct fields,
// an interface method set, or a defined/enum type with its constants.
func (m *pkgModel) typeDecl(ti *typeInfo) string {
	switch t := ti.spec.Type.(type) {
	case *ast.StructType:
		return m.renderStructDecl(ti.name, t)
	case *ast.InterfaceType:
		return codeBlock(m.printNode(&ast.GenDecl{Tok: token.TYPE, Specs: []ast.Spec{ti.spec}}))
	default:
		var b strings.Builder
		b.WriteString(codeBlock("type " + ti.name + " " + m.printNode(ti.spec.Type)))
		if len(ti.consts) > 0 {
			b.WriteString("\nThe possible values are:\n\n")
			for _, c := range ti.consts {
				if c.value != "" {
					fmt.Fprintf(&b, "- `%s` = `%s`", c.name, c.value)
				} else {
					fmt.Fprintf(&b, "- `%s`", c.name)
				}
				if c.doc != "" {
					fmt.Fprintf(&b, " — %s", c.doc)
				}
				b.WriteString("\n")
			}
		}
		return b.String()
	}
}

// renderStructDecl prints `type Name struct { ... }` with only exported fields,
// keeping each field's trailing/leading doc comment as an inline comment.
func (m *pkgModel) renderStructDecl(name string, st *ast.StructType) string {
	var lines []string
	lines = append(lines, fmt.Sprintf("type %s struct {", name))
	for _, field := range st.Fields.List {
		if len(field.Names) == 0 {
			continue // embedded field
		}
		var names []string
		for _, n := range field.Names {
			if isExported(n.Name) {
				names = append(names, n.Name)
			}
		}
		if len(names) == 0 {
			continue
		}
		line := "\t" + strings.Join(names, ", ") + " " + m.printNode(field.Type)
		if c := oneLine(fieldDoc(field)); c != "" {
			line += " // " + c
		}
		lines = append(lines, line)
	}
	lines = append(lines, "}")
	src := strings.Join(lines, "\n")
	if formatted, err := format.Source([]byte(src)); err == nil {
		src = string(formatted)
	}
	return codeBlock(src)
}

// renderServiceMethods renders one section per interface method, taking the
// signature from the interface and the doc comment from the matching method on
// the unexported *ServiceImpl type. caption, when non-empty, is shown under
// every method heading to name the call path (e.g. via client.Volumes); it is
// empty for companion namespaces, where the section heading already conveys the
// access path.
func (m *pkgModel) renderServiceMethods(b *strings.Builder, svc *typeInfo, depth int, caption string) {
	it, ok := svc.spec.Type.(*ast.InterfaceType)
	if !ok {
		return
	}
	implName := lowerFirst(svc.name) + "Impl"
	for _, field := range it.Methods.List {
		ft, ok := field.Type.(*ast.FuncType)
		if !ok || len(field.Names) == 0 {
			continue // embedded interface
		}
		name := field.Names[0].Name
		if !isExported(name) {
			continue
		}
		doc := field.Doc.Text()
		if doc == "" {
			if impl := m.methods[implName][name]; impl != nil {
				doc = impl.Doc.Text()
			}
		}
		m.renderSection(b, depth, name, caption, m.methodSig(name, ft), doc, ft)
	}
}

// renderObjectMethods renders one section per exported method declared on the
// type itself (pointer or value receiver). These carry no call-path caption:
// they are plain methods on the type the page documents. Only the merged
// service methods, which are reached through a Client field rather than the
// value itself, are called out (see renderServiceMethods).
func (m *pkgModel) renderObjectMethods(b *strings.Builder, typeName string, depth int) {
	methods := m.methods[typeName]
	names := make([]string, 0, len(methods))
	for name := range methods {
		if isExported(name) {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	for _, name := range names {
		fd := methods[name]
		m.renderSection(b, depth, name, "", m.methodSig(name, fd.Type), fd.Doc.Text(), fd.Type)
	}
}

// serviceAccessCaption is the caption shown under a merged service method,
// naming the Client field that exposes the service (e.g. client.Volumes). It is
// empty when the service is not reached through the Client struct. The caption
// is a sentence fragment, so it takes no trailing period.
func (m *pkgModel) serviceAccessCaption(svcName string) string {
	field := m.serviceAccessors[svcName]
	if field == "" {
		return ""
	}
	return fmt.Sprintf("_Accessed via `client.%s`_", field)
}

// renderFunc renders a section for a package-level function (constructor or
// standalone), keeping the `func` keyword to mark it as package-level.
func (m *pkgModel) renderFunc(b *strings.Builder, fd *ast.FuncDecl, depth int) {
	clone := *fd
	clone.Body = nil
	clone.Doc = nil
	m.renderSection(b, depth, fd.Name.Name, "", m.printNode(&clone), fd.Doc.Text(), fd.Type)
}

// renderSection writes a heading (at the given depth), an optional caption
// naming how the callable is reached, the signature code block, the doc
// comment, and the folded *Params field list (if the final argument is one).
func (m *pkgModel) renderSection(b *strings.Builder, depth int, name, caption, signature, doc string, ft *ast.FuncType) {
	fmt.Fprintf(b, "%s %s\n\n", strings.Repeat("#", depth), name)
	if caption != "" {
		b.WriteString(caption)
		b.WriteString("\n\n")
	}
	b.WriteString(codeBlock(signature))
	b.WriteString("\n")
	writeDoc(b, doc)
	if pt := lastParamsType(ft); pt != "" {
		m.renderParams(b, pt)
	}
}

// renderParams documents a method's *XxxParams options struct: the type name,
// its doc comment, and one Markdown list item per exported field. The struct is
// always shown when a method takes one, even with no fields (e.g. struct{}), so
// the reader sees which options type the method accepts.
func (m *pkgModel) renderParams(b *strings.Builder, typeName string) {
	fmt.Fprintf(b, "**Parameters** (`%s`)\n\n", typeName)

	if ti := m.types[typeName]; ti != nil {
		writeDoc(b, ti.doc)
	}

	var lines []string
	if st := m.resolveStruct(typeName); st != nil {
		for _, field := range st.Fields.List {
			if len(field.Names) == 0 {
				continue
			}
			typeStr := m.printNode(field.Type)
			doc := godocRefsToCode(oneLine(fieldDoc(field)))
			for _, n := range field.Names {
				if !isExported(n.Name) {
					continue
				}
				line := fmt.Sprintf("- `%s` (`%s`)", n.Name, typeStr)
				if doc != "" {
					line += ": " + doc
				}
				lines = append(lines, line)
			}
		}
	}

	if len(lines) > 0 {
		b.WriteString(strings.Join(lines, "\n"))
		b.WriteString("\n\n")
	} else {
		b.WriteString("_No configurable options._\n\n")
	}
}

// resolveStruct returns the struct type for typeName, following at most a chain
// of single-identifier type aliases (e.g. SidecarExecParams -> SandboxExecParams).
func (m *pkgModel) resolveStruct(typeName string) *ast.StructType {
	seen := map[string]bool{}
	for typeName != "" && !seen[typeName] {
		seen[typeName] = true
		ti := m.types[typeName]
		if ti == nil {
			return nil
		}
		switch t := ti.spec.Type.(type) {
		case *ast.StructType:
			return t
		case *ast.Ident:
			typeName = t.Name
		default:
			return nil
		}
	}
	return nil
}

// introDoc is the Go SDK reference landing page, written to intro.md alongside
// the generated per-type pages. It is authored as a sibling intro.md and
// embedded at build time (so the command stays self-contained). The Python SDK
// and CLI references likewise emit their intro into the generated output, so the
// frontend treats every reference surface's landing page uniformly as generated
// Markdown (the empty URL id resolves to intro.md). The title is derived from
// the first heading; the description frontmatter feeds OpenGraph metadata.
//
//go:embed intro.md
var introDoc string

func writeOutput(outDir string, m *pkgModel, pages []*page) error {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}

	// The reference landing page, generated as intro.md alongside the per-type
	// pages; the empty URL id resolves to it.
	if err := os.WriteFile(filepath.Join(outDir, "intro.md"), []byte(introDoc), 0o644); err != nil {
		return err
	}

	type sidebarItem struct {
		Label    string `json:"label"`
		Category string `json:"category"`
	}
	var items []sidebarItem

	// Pages are written flat (one file per type), mirroring the Python pydoc
	// output; the type name is both the filename stem and the URL id segment.
	for _, p := range pages {
		content := m.render(p)
		fname := filepath.Join(outDir, p.label+".md")
		if err := os.WriteFile(fname, []byte(content), 0o644); err != nil {
			return err
		}
		items = append(items, sidebarItem{Label: p.label, Category: p.category()})
	}

	data, err := json.Marshal(struct {
		Items []sidebarItem `json:"items"`
	}{Items: items})
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(outDir, "sidebar.json"), data, 0o644)
}

// --- small helpers ---

func (m *pkgModel) printNode(n ast.Node) string {
	var buf bytes.Buffer
	cfg := printer.Config{Mode: printer.UseSpaces | printer.TabIndent, Tabwidth: 4}
	if err := cfg.Fprint(&buf, m.fset, n); err != nil {
		return ""
	}
	return buf.String()
}

// methodSig renders a method signature without the `func` keyword or receiver,
// e.g. "FromName(ctx context.Context, name string, params *VolumeFromNameParams) (*Volume, error)".
func (m *pkgModel) methodSig(name string, ft *ast.FuncType) string {
	fd := &ast.FuncDecl{Name: ast.NewIdent(name), Type: ft}
	return strings.TrimPrefix(m.printNode(fd), "func ")
}

func codeBlock(s string) string {
	return "```go\n" + strings.TrimRight(s, "\n") + "\n```\n"
}

// godocRefRe matches a godoc symbol reference: a bracketed Go identifier,
// optionally qualified (time.Duration), a pointer (*Sandbox), or a method
// (Sandbox.Detach). Multi-word bracketed text — e.g. godoc's "[Go blog]: URL"
// link definitions — contains spaces and so never matches.
var godocRefRe = regexp.MustCompile(`\[(\*?[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\]`)

// godocRefsToCode rewrites godoc symbol references into Markdown code spans, so
// [Sandbox.Detach] renders as `Sandbox.Detach`. The docs site does not
// understand godoc's bracket-link syntax; without this the brackets render
// literally. Only used on prose (doc comments), never on code blocks.
func godocRefsToCode(s string) string {
	return godocRefRe.ReplaceAllString(s, "`$1`")
}

// writeDoc writes a doc comment as a prose block (followed by a blank line),
// converting godoc symbol references into code spans. It is a no-op when the
// comment is empty.
func writeDoc(b *strings.Builder, doc string) {
	if doc = strings.TrimSpace(godocRefsToCode(doc)); doc != "" {
		b.WriteString(doc)
		b.WriteString("\n\n")
	}
}

// fieldDoc returns a struct field's documentation, preferring a leading doc
// comment over a trailing line comment.
func fieldDoc(field *ast.Field) string {
	if field.Doc != nil {
		return field.Doc.Text()
	}
	if field.Comment != nil {
		return field.Comment.Text()
	}
	return ""
}

// lastParamsType returns the bare name of the final parameter's type when it is
// a pointer to a *XxxParams struct, else "".
func lastParamsType(ft *ast.FuncType) string {
	if ft.Params == nil || len(ft.Params.List) == 0 {
		return ""
	}
	last := ft.Params.List[len(ft.Params.List)-1]
	star, ok := last.Type.(*ast.StarExpr)
	if !ok {
		return ""
	}
	id, ok := star.X.(*ast.Ident)
	if !ok || !isParamsType(id.Name) {
		return ""
	}
	return id.Name
}

func recvTypeName(recv *ast.FieldList) string {
	if recv == nil || len(recv.List) == 0 {
		return ""
	}
	t := recv.List[0].Type
	if star, ok := t.(*ast.StarExpr); ok {
		t = star.X
	}
	switch e := t.(type) {
	case *ast.Ident:
		return e.Name
	case *ast.IndexExpr: // generic receiver, e.g. Foo[T]
		if id, ok := e.X.(*ast.Ident); ok {
			return id.Name
		}
	}
	return ""
}

func firstResultBaseType(fd *ast.FuncDecl) string {
	if fd.Type.Results == nil || len(fd.Type.Results.List) == 0 {
		return ""
	}
	t := fd.Type.Results.List[0].Type
	if star, ok := t.(*ast.StarExpr); ok {
		t = star.X
	}
	if id, ok := t.(*ast.Ident); ok {
		return id.Name
	}
	return ""
}

func isServiceInterface(ti *typeInfo) bool {
	if !strings.HasSuffix(ti.name, "Service") {
		return false
	}
	_, ok := ti.spec.Type.(*ast.InterfaceType)
	return ok
}

func isParamsType(name string) bool { return strings.HasSuffix(name, "Params") }

// baseTypeName returns the underlying identifier name of a (possibly pointer)
// type expression, or "" for anything else (e.g. a qualified io.WriteCloser).
func baseTypeName(expr ast.Expr) string {
	switch e := expr.(type) {
	case *ast.StarExpr:
		return baseTypeName(e.X)
	case *ast.Ident:
		return e.Name
	}
	return ""
}

// hasExportedMethods reports whether the named type declares any exported method.
func (m *pkgModel) hasExportedMethods(typeName string) bool {
	for name := range m.methods[typeName] {
		if isExported(name) {
			return true
		}
	}
	return false
}

// isErrorType reports whether the named type declares an Error() method, i.e.
// implements the error interface. This catches error types regardless of name
// (e.g. InternalFailure, which does not end in "Error").
func (m *pkgModel) isErrorType(name string) bool {
	_, ok := m.methods[name]["Error"]
	return ok
}

func isExported(name string) bool {
	if name == "" {
		return false
	}
	r := []rune(name)[0]
	return unicode.IsUpper(r)
}

func lowerFirst(s string) string {
	if s == "" {
		return s
	}
	r := []rune(s)
	r[0] = unicode.ToLower(r[0])
	return string(r)
}

// oneLine collapses a doc comment into a single trimmed line for inline use.
func oneLine(s string) string {
	return strings.TrimSpace(strings.Join(strings.Fields(s), " "))
}
