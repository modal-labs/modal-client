// Command gen_reference_docs generates Markdown reference documentation for
// the public JS SDK in client/js (the `modal` npm package).
//
// It mirrors the Go reference pipeline (client/go/modal_docs): one Markdown
// page per primary type, written so the docs site can render them under
// /docs/sdk/js/latest/<Type>.
//
// The public surface is exactly the set of names re-exported from
// src/index.ts; everything else in the source modules (internal helpers,
// proto builders) is ignored. Four SDK conventions shape the output:
//
//   - Each resource exposes both an entity class (e.g. Volume) and a service
//     class (e.g. VolumeService, reached via modal.volumes). The service
//     class's methods are merged onto the entity page, so a reader sees
//     fromName/ephemeral/delete next to Volume's own methods.
//   - Every XxxParams options type is folded into the page section of the
//     method whose final argument is XxxParams, rather than getting a page of
//     its own.
//   - Classes whose prototype chain reaches Error are collected onto a single
//     Errors page rather than getting a page each.
//   - A "companion" class reached through a property or getter on a primary
//     class (e.g. Sandbox.filesystem -> SandboxFilesystem) is folded into the
//     primary's page as a namespace section rather than getting a page of its
//     own — the JS analog of Python's mdmd:namespace folding.
//
// Usage:
//
//	npx tsx modal_docs/gen_reference_docs.ts <output_dir> [<source_dir>]
//
// <source_dir> defaults to "src", which under `npx tsx modal_docs/...` (cwd
// client/js) is the package source directory. Pages are written flat to
// <output_dir>/<Type>.md, a landing page to <output_dir>/intro.md, and a
// sidebar index to <output_dir>/sidebar.json.

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import ts from "typescript";

// hiddenSymbols are names on the public export surface that are not
// user-facing API and should not be documented (typically validation helpers
// that must be importable by other SDK-adjacent packages).
const hiddenSymbols = new Set(["checkForRenamedParams"]);

// clientTypeName is the exported class whose properties expose the service
// classes (the SDK entrypoint).
const clientTypeName = "ModalClient";

// errorsPageLabel is the title/URL stem of the aggregate page that collects
// all error types, which are too numerous and small to warrant a page each.
const errorsPageLabel = "Errors";

// --- doc comment parsing ---

/** The parsed contents of one JSDoc block. */
export interface DocInfo {
  /** Prose paragraphs before the first block tag. */
  summary: string;
  /** `@param name - text` entries, in order. */
  params: { name: string; text: string }[];
  /** `@returns` text, if present. */
  returns?: string;
  /** `@throws {Type} text` entries, in order. */
  throws: { type: string; text: string }[];
  /** `@example` bodies (usually containing a fenced code block), verbatim. */
  examples: string[];
  /** `@deprecated` text, if the tag is present (may be empty). */
  deprecated?: string;
  /** True when tagged `@hidden`, `@ignore`, or `@internal`. */
  hidden: boolean;
}

function emptyDoc(): DocInfo {
  return { summary: "", params: [], throws: [], examples: [], hidden: false };
}

/**
 * Parses the raw text of a JSDoc block into its summary and the block tags
 * the docs render. Unknown tags are dropped. Inline tags like {@link} are
 * left in place and rewritten at render time.
 */
export function parseDocComment(raw: string): DocInfo {
  const doc = emptyDoc();
  const body = raw
    .replace(/^\/\*\*[ \t]*/, "")
    .replace(/\s*\*\/$/, "")
    .split("\n")
    .map((line) => line.replace(/^\s*\* ?/, ""))
    .join("\n");

  type Section = { tag: string; lines: string[] };
  const sections: Section[] = [{ tag: "", lines: [] }];
  for (const line of body.split("\n")) {
    const tagMatch = /^@([A-Za-z]+)\b\s*(.*)$/.exec(line.trim());
    if (tagMatch) {
      sections.push({ tag: tagMatch[1], lines: [tagMatch[2]] });
    } else {
      sections[sections.length - 1].lines.push(line);
    }
  }

  for (const section of sections) {
    const text = section.lines.join("\n").trim();
    switch (section.tag) {
      case "":
        doc.summary = text;
        break;
      case "param": {
        const m = /^([A-Za-z_$][\w$]*)\s*(?:-\s*)?([\s\S]*)$/.exec(text);
        if (m) doc.params.push({ name: m[1], text: m[2].trim() });
        break;
      }
      case "returns":
      case "return":
        doc.returns = text;
        break;
      case "throws":
      case "throw": {
        // The raised type may be written JSDoc-style (`@throws {Type} desc`)
        // or TSDoc-style (`@throws {@link Type} desc`); extract the bare name
        // from either form.
        const link =
          /^\{@link(?:code|plain)?\s+([^}|\s]+)\s*(?:\|\s*)?([^}]*)\}\s*([\s\S]*)$/.exec(
            text,
          );
        if (link) {
          const type = link[2].trim() || link[1].replace(/#/g, ".");
          doc.throws.push({ type, text: link[3].trim() });
          break;
        }
        const m = /^\{([^}]+)\}\s*([\s\S]*)$/.exec(text);
        if (m) doc.throws.push({ type: m[1], text: m[2].trim() });
        else doc.throws.push({ type: "", text });
        break;
      }
      case "example":
        doc.examples.push(text);
        break;
      case "deprecated":
        doc.deprecated = text;
        break;
      case "hidden":
      case "ignore":
      case "internal":
        doc.hidden = true;
        break;
      default:
        break; // unknown tag: dropped
    }
  }
  return doc;
}

/** Returns the parsed JSDoc block immediately preceding a node. */
function docFor(node: ts.Node, sf: ts.SourceFile): DocInfo {
  const jsDocs = ts.getJSDocCommentsAndTags(node).filter(ts.isJSDoc);
  if (jsDocs.length === 0) return emptyDoc();
  const last = jsDocs[jsDocs.length - 1];
  return parseDocComment(sf.text.slice(last.getStart(sf), last.getEnd()));
}

// linkRe matches TSDoc inline link tags: {@link Target}, {@link Target label},
// or {@link Target | label}, plus the {@linkcode}/{@linkplain} variants.
const linkRe = /\{@link(?:code|plain)?\s+([^}|\s]+)\s*(?:\|\s*)?([^}]*)\}/g;

/**
 * Rewrites TSDoc {@link} tags into Markdown code spans, so
 * `{@link Sandbox#exec Sandbox.exec()}` renders as `Sandbox.exec()`. The docs
 * site does not resolve TSDoc link syntax; without this the braces render
 * literally. Only used on prose, never on code blocks.
 */
export function linkRefsToCode(s: string): string {
  return s.replace(linkRe, (_all, target: string, label: string) => {
    const text = label.trim() || target.replace(/#/g, ".");
    return "`" + text + "`";
  });
}

// --- model building ---

/** One public method (or constructor / free function) ready to render. */
interface MethodInfo {
  name: string;
  /** Signature text, sliced from the source (no body). */
  signature: string;
  doc: DocInfo;
  /** Name of the trailing XxxParams type folded into this section, if any. */
  paramsTypeName?: string;
  /** Name of the final parameter (used to drop its redundant `@param`). */
  lastParamName?: string;
  order: number;
}

/** One public property or getter on a class, for the declaration block. */
interface PropInfo {
  name: string;
  /** The rendered line inside the class declaration block. */
  declLine: string;
  doc: DocInfo;
  /** Bare name of the property's declared type (generics stripped). */
  baseTypeName?: string;
}

/** A parsed public type declaration (class, interface, type alias, or enum). */
interface TypeDeclInfo {
  name: string;
  kind: "class" | "interface" | "typeAlias" | "enum";
  doc: DocInfo;
  node: ts.Declaration;
  sf: ts.SourceFile;
  methods: MethodInfo[]; // public instance methods
  staticMethods: MethodInfo[];
  ctor?: MethodInfo; // public, non-@ignore constructor
  properties: PropInfo[];
  /** Bare name of the extends clause target, if any. */
  extendsName?: string;
}

/** The parsed view of the package needed for doc generation. */
export interface PkgModel {
  /** Every exported type declaration in the public modules, keyed by name. */
  types: Map<string, TypeDeclInfo>;
  /** Every exported top-level function in the public modules, keyed by name. */
  funcs: Map<string, MethodInfo>;
  /** Names re-exported from index.ts (the public surface). */
  publicNames: Set<string>;
  /**
   * Maps a service class name to the property on ModalClient that exposes it
   * (e.g. "VolumeService" -> "volumes"), so docs can name the real call path
   * (modal.volumes.fromName). The mapping is not mechanical (ClsService ->
   * cls, FunctionCallService -> functionCalls), so it is read from the class
   * rather than derived from the type name.
   */
  serviceAccessors: Map<string, string>;
}

/**
 * Builds the package model from in-memory sources keyed by module filename
 * (e.g. "index.ts", "sandbox.ts"). The set of modules parsed and the public
 * surface both derive from the re-export declarations in "index.ts".
 */
export function buildModel(files: Record<string, string>): PkgModel {
  const indexSource = files["index.ts"];
  if (indexSource === undefined) {
    throw new Error("no index.ts in source set");
  }
  const indexSf = ts.createSourceFile(
    "index.ts",
    indexSource,
    ts.ScriptTarget.ES2022,
    true,
  );

  const publicNames = new Set<string>();
  const moduleNames = new Set<string>();
  for (const stmt of indexSf.statements) {
    if (!ts.isExportDeclaration(stmt)) continue;
    if (!stmt.moduleSpecifier || !ts.isStringLiteral(stmt.moduleSpecifier)) {
      continue;
    }
    if (!stmt.exportClause || !ts.isNamedExports(stmt.exportClause)) continue;
    moduleNames.add(stmt.moduleSpecifier.text.replace(/^\.\//, "") + ".ts");
    for (const spec of stmt.exportClause.elements) {
      publicNames.add(spec.name.text);
    }
  }

  const model: PkgModel = {
    types: new Map(),
    funcs: new Map(),
    publicNames,
    serviceAccessors: new Map(),
  };

  for (const moduleName of moduleNames) {
    const source = files[moduleName];
    if (source === undefined) {
      throw new Error(
        `module ${moduleName} re-exported from index.ts not found`,
      );
    }
    const sf = ts.createSourceFile(
      moduleName,
      source,
      ts.ScriptTarget.ES2022,
      true,
    );
    collectModule(model, sf);
  }

  buildServiceAccessors(model);
  return model;
}

/** Reads the source modules from disk and builds the model. */
export function buildModelFromDir(srcDir: string): PkgModel {
  const files: Record<string, string> = {
    "index.ts": fs.readFileSync(path.join(srcDir, "index.ts"), "utf8"),
  };
  const indexSf = ts.createSourceFile(
    "index.ts",
    files["index.ts"],
    ts.ScriptTarget.ES2022,
    true,
  );
  for (const stmt of indexSf.statements) {
    if (!ts.isExportDeclaration(stmt)) continue;
    if (!stmt.moduleSpecifier || !ts.isStringLiteral(stmt.moduleSpecifier)) {
      continue;
    }
    const moduleName = stmt.moduleSpecifier.text.replace(/^\.\//, "") + ".ts";
    files[moduleName] = fs.readFileSync(path.join(srcDir, moduleName), "utf8");
  }
  return buildModel(files);
}

/** Collects every exported top-level declaration in one module. */
function collectModule(model: PkgModel, sf: ts.SourceFile): void {
  for (const stmt of sf.statements) {
    if (!isExportedStatement(stmt)) continue;
    if (ts.isClassDeclaration(stmt) && stmt.name) {
      model.types.set(stmt.name.text, collectClass(stmt, sf));
    } else if (ts.isInterfaceDeclaration(stmt)) {
      model.types.set(stmt.name.text, {
        name: stmt.name.text,
        kind: "interface",
        doc: docFor(stmt, sf),
        node: stmt,
        sf,
        methods: [],
        staticMethods: [],
        properties: [],
      });
    } else if (ts.isTypeAliasDeclaration(stmt)) {
      model.types.set(stmt.name.text, {
        name: stmt.name.text,
        kind: "typeAlias",
        doc: docFor(stmt, sf),
        node: stmt,
        sf,
        methods: [],
        staticMethods: [],
        properties: [],
      });
    } else if (ts.isEnumDeclaration(stmt)) {
      model.types.set(stmt.name.text, {
        name: stmt.name.text,
        kind: "enum",
        doc: docFor(stmt, sf),
        node: stmt,
        sf,
        methods: [],
        staticMethods: [],
        properties: [],
      });
    } else if (ts.isFunctionDeclaration(stmt) && stmt.name) {
      model.funcs.set(stmt.name.text, {
        name: stmt.name.text,
        signature: signatureText(stmt, sf),
        doc: docFor(stmt, sf),
        paramsTypeName: lastParamsTypeName(stmt),
        lastParamName: lastParamName(stmt),
        order: 0,
      });
    }
  }
}

function isExportedStatement(stmt: ts.Statement): boolean {
  const modifiers = ts.canHaveModifiers(stmt)
    ? ts.getModifiers(stmt)
    : undefined;
  return !!modifiers?.some((m) => m.kind === ts.SyntaxKind.ExportKeyword);
}

/** Collects the public members of a class declaration. */
function collectClass(
  cls: ts.ClassDeclaration,
  sf: ts.SourceFile,
): TypeDeclInfo {
  const info: TypeDeclInfo = {
    name: cls.name!.text,
    kind: "class",
    doc: docFor(cls, sf),
    node: cls,
    sf,
    methods: [],
    staticMethods: [],
    properties: [],
    extendsName: heritageName(cls),
  };

  // Overload declarations are grouped under one MethodInfo per method name.
  const methodDecls = new Map<string, ts.MethodDeclaration[]>();

  let order = 0;
  for (const member of cls.members) {
    if (!isPublicMember(member, sf)) continue;
    if (ts.isMethodDeclaration(member) && ts.isIdentifier(member.name)) {
      const name = member.name.text;
      if (!methodDecls.has(name)) methodDecls.set(name, []);
      methodDecls.get(name)!.push(member);
    } else if (
      ts.isPropertyDeclaration(member) &&
      ts.isIdentifier(member.name)
    ) {
      info.properties.push(propertyInfo(member, sf));
    } else if (
      ts.isGetAccessorDeclaration(member) &&
      ts.isIdentifier(member.name)
    ) {
      info.properties.push(getterInfo(member, sf));
    } else if (ts.isConstructorDeclaration(member)) {
      // Constructor parameter properties (e.g. `public host: string`) are
      // public fields and belong in the declaration block.
      for (const param of member.parameters) {
        if (
          ts.getModifiers(param)?.length &&
          ts.isIdentifier(param.name) &&
          isPublicName(param.name.text)
        ) {
          info.properties.push(paramPropertyInfo(param, sf));
        }
      }
      info.ctor = {
        name: "constructor",
        signature: signatureText(member, sf).replace(
          /^constructor/,
          `new ${info.name}`,
        ),
        doc: docFor(member, sf),
        paramsTypeName: lastParamsTypeName(member),
        lastParamName: lastParamName(member),
        order,
      };
    }
    order++;
  }

  for (const [name, decls] of methodDecls) {
    const isStatic = !!ts
      .getModifiers(decls[0])
      ?.some((m) => m.kind === ts.SyntaxKind.StaticKeyword);
    const method = methodInfo(name, decls, sf);
    method.order = info.methods.length + info.staticMethods.length;
    (isStatic ? info.staticMethods : info.methods).push(method);
  }
  return info;
}

/** Builds one MethodInfo from a method's declarations (overloads + impl). */
function methodInfo(
  name: string,
  decls: ts.MethodDeclaration[],
  sf: ts.SourceFile,
): MethodInfo {
  // When a method has overloads, the overload signatures are the callable
  // surface; the implementation signature is shown only when it stands alone.
  const overloads = decls.filter((d) => !d.body);
  const shown = overloads.length > 0 ? overloads : decls;
  const signature = shown.map((d) => signatureText(d, sf)).join("\n");

  const documented = decls.find(
    (d) => ts.getJSDocCommentsAndTags(d).filter(ts.isJSDoc).length > 0,
  );
  const doc = documented ? docFor(documented, sf) : emptyDoc();

  // The params type is usually declared on the implementation signature (the
  // overloads narrow it), so scan implementation-last.
  let paramsTypeName: string | undefined;
  let lastName: string | undefined;
  for (const d of [...decls].reverse()) {
    paramsTypeName = lastParamsTypeName(d);
    lastName = lastParamName(d);
    if (paramsTypeName) break;
  }
  return {
    name,
    signature,
    doc,
    paramsTypeName,
    lastParamName: lastName,
    order: 0,
  };
}

function propertyInfo(
  prop: ts.PropertyDeclaration,
  sf: ts.SourceFile,
): PropInfo {
  const readonly = ts
    .getModifiers(prop)
    ?.some((m) => m.kind === ts.SyntaxKind.ReadonlyKeyword);
  const name = (prop.name as ts.Identifier).text;
  let line = (readonly ? "readonly " : "") + name;
  if (prop.questionToken) line += "?";
  if (prop.type) line += ": " + oneLine(prop.type.getText(sf));
  line += ";";
  return {
    name,
    declLine: line,
    doc: docFor(prop, sf),
    baseTypeName: prop.type ? baseTypeName(prop.type) : undefined,
  };
}

function getterInfo(
  getter: ts.GetAccessorDeclaration,
  sf: ts.SourceFile,
): PropInfo {
  const name = (getter.name as ts.Identifier).text;
  let line = `get ${name}()`;
  if (getter.type) line += ": " + oneLine(getter.type.getText(sf));
  line += ";";
  return {
    name,
    declLine: line,
    doc: docFor(getter, sf),
    baseTypeName: getter.type ? baseTypeName(getter.type) : undefined,
  };
}

function paramPropertyInfo(
  param: ts.ParameterDeclaration,
  sf: ts.SourceFile,
): PropInfo {
  const readonly = ts
    .getModifiers(param)
    ?.some((m) => m.kind === ts.SyntaxKind.ReadonlyKeyword);
  const name = (param.name as ts.Identifier).text;
  let line = (readonly ? "readonly " : "") + name;
  if (param.questionToken) line += "?";
  if (param.type) line += ": " + oneLine(param.type.getText(sf));
  line += ";";
  return {
    name,
    declLine: line,
    doc: docFor(param, sf),
    baseTypeName: param.type ? baseTypeName(param.type) : undefined,
  };
}

/**
 * Reports whether a class member is part of the public API surface: not
 * private (by modifier, `#` name, or `_` prefix) and not tagged
 * `@hidden`/`@ignore`/`@internal`.
 */
function isPublicMember(member: ts.ClassElement, sf: ts.SourceFile): boolean {
  const modifiers = ts.canHaveModifiers(member)
    ? ts.getModifiers(member)
    : undefined;
  if (
    modifiers?.some(
      (m) =>
        m.kind === ts.SyntaxKind.PrivateKeyword ||
        m.kind === ts.SyntaxKind.ProtectedKeyword,
    )
  ) {
    return false;
  }
  if (member.name) {
    if (ts.isPrivateIdentifier(member.name)) return false;
    if (ts.isIdentifier(member.name) && !isPublicName(member.name.text)) {
      return false;
    }
  }
  return !docFor(member, sf).hidden;
}

function isPublicName(name: string): boolean {
  return !name.startsWith("_") && !name.startsWith("#");
}

/**
 * Records, for each service class exposed as a property on ModalClient, the
 * property name that reaches it (e.g. modal.volumes -> VolumeService). Doc
 * rendering uses this to show the real call path for merged service methods.
 */
function buildServiceAccessors(model: PkgModel): void {
  const client = model.types.get(clientTypeName);
  if (!client) return;
  for (const prop of client.properties) {
    const typeName = prop.baseTypeName;
    if (typeName && isServiceClass(model.types.get(typeName))) {
      model.serviceAccessors.set(typeName, prop.name);
    }
  }
}

function isServiceClass(ti: TypeDeclInfo | undefined): boolean {
  return !!ti && ti.kind === "class" && ti.name.endsWith("Service");
}

function isParamsType(name: string): boolean {
  return name.endsWith("Params");
}

/**
 * Reports whether the named class's heritage chain reaches Error, i.e. it is
 * a throwable error type. This catches error types regardless of name (e.g.
 * InternalFailure, which does not end in "Error").
 */
function isErrorType(model: PkgModel, name: string): boolean {
  const seen = new Set<string>();
  let current: string | undefined = name;
  while (current && !seen.has(current)) {
    seen.add(current);
    const ti = model.types.get(current);
    if (!ti || ti.kind !== "class") return false;
    if (ti.extendsName === "Error") return true;
    current = ti.extendsName;
  }
  return false;
}

// --- page building ---

/** A dependent class folded into a primary class's page. */
interface Companion {
  propName: string; // property/getter on the primary, e.g. "filesystem"
  propDoc: DocInfo; // doc comment on that property/getter
  ti: TypeDeclInfo;
  isService: boolean; // the companion is a service class
}

/** A single output Markdown page for one public type (or a free function). */
export interface Page {
  label: string; // page title and filename stem, e.g. "Volume"
  ti?: TypeDeclInfo;
  /**
   * When set, a service class whose methods are merged onto this page (e.g.
   * VolumeService merged onto the Volume page).
   */
  service?: TypeDeclInfo;
  freeFunc?: MethodInfo;
  /**
   * When non-empty, makes this an aggregate page that documents several small
   * related types as sections (used for the Errors page).
   */
  groupedTypes?: TypeDeclInfo[];
  companions: Companion[];
}

/**
 * Returns the page label for a type name. The entity class for functions is
 * named Function_ to avoid shadowing the JS builtin; its page (like the
 * identifier users see in Python and Go) is plain Function.
 */
export function pageLabel(name: string): string {
  return name.replace(/_$/, "");
}

function pageCategory(p: Page): string {
  return p.freeFunc ? "function" : "type";
}

/** Classifies every public declaration into the set of pages. */
export function buildPages(model: PkgModel): Page[] {
  // Keyed by type name (not label), so service matching can find Function_.
  const pages = new Map<string, Page>();
  const errorTypes: TypeDeclInfo[] = [];

  // Entity/value types get their own page. Params types are folded into
  // methods; service classes are handled separately below; error types are
  // collected onto a single aggregate page.
  for (const [name, ti] of model.types) {
    if (!model.publicNames.has(name) || hiddenSymbols.has(name)) continue;
    if (ti.doc.hidden) continue;
    if (isParamsType(name) || isServiceClass(ti)) continue;
    if (isErrorType(model, name)) {
      errorTypes.push(ti);
      continue;
    }
    pages.set(name, { label: pageLabel(name), ti, companions: [] });
  }

  // Merge each service class into its matching entity page, or give it a page
  // of its own when there is no match (it may still be folded as a companion
  // below, e.g. SidecarService reached via Sandbox.experimentalSidecars).
  for (const [name, ti] of model.types) {
    if (!model.publicNames.has(name) || !isServiceClass(ti)) continue;
    if (ti.doc.hidden) continue;
    const target = name.slice(0, -"Service".length);
    const entity = pages.get(target) ?? pages.get(target + "_");
    if (entity) {
      entity.service = ti;
    } else {
      pages.set(name, {
        label: pageLabel(name),
        ti,
        service: ti,
        companions: [],
      });
    }
  }

  foldCompanions(model, pages);

  // Free functions (rare) get their own page.
  for (const [name, fn] of model.funcs) {
    if (!model.publicNames.has(name) || hiddenSymbols.has(name)) continue;
    if (fn.doc.hidden) continue;
    pages.set(name, { label: name, freeFunc: fn, companions: [] });
  }

  if (errorTypes.length > 0) {
    errorTypes.sort((a, b) => a.name.localeCompare(b.name));
    pages.set(errorsPageLabel, {
      label: errorsPageLabel,
      groupedTypes: errorTypes,
      companions: [],
    });
  }

  const out = [...pages.values()];
  out.sort((a, b) => a.label.localeCompare(b.label));
  for (const p of out) {
    p.companions.sort((a, b) => a.propName.localeCompare(b.propName));
  }
  return out;
}

/**
 * Detects dependent classes reached through a property or getter on a primary
 * class (e.g. Sandbox.filesystem -> SandboxFilesystem) and folds them into the
 * primary's page as namespace sections, removing their standalone page. A
 * class qualifies when it is referenced via a public property of a page class
 * and is either a service class or has public methods of its own (a concrete
 * operations namespace) — plain data types are left as their own pages. A
 * companion reached from several primaries (e.g. both Sandbox and
 * SidecarContainer expose .filesystem) is documented on each so every page is
 * self-contained.
 */
function foldCompanions(model: PkgModel, pages: Map<string, Page>): void {
  const additions: { owner: string; comp: Companion }[] = [];
  const deletions = new Set<string>();

  for (const [name, p] of pages) {
    if (!p.ti || p.ti.kind !== "class") continue;
    for (const prop of p.ti.properties) {
      const dep = prop.baseTypeName;
      if (!dep || dep === name) continue;
      const depPage = pages.get(dep);
      if (!depPage) continue; // not a standalone page (external, merged, grouped)
      const ti = model.types.get(dep)!;
      if (ti.kind !== "class") continue; // interfaces stay standalone pages
      const isSvc = isServiceClass(ti);
      if (!isSvc && ti.methods.length === 0) continue; // plain data class
      additions.push({
        owner: name,
        comp: { propName: prop.name, propDoc: prop.doc, ti, isService: isSvc },
      });
      deletions.add(dep);
    }
  }

  for (const dep of deletions) pages.delete(dep);
  for (const a of additions) {
    // Skip if the owner was itself folded away into another type.
    const owner = pages.get(a.owner);
    if (owner) owner.companions.push(a.comp);
  }
}

// --- rendering ---

/** Produces the full Markdown for a page. */
export function renderPage(model: PkgModel, p: Page): string {
  const b: string[] = [];
  b.push(`# ${p.label}\n\n`);

  if (p.groupedTypes?.length) {
    for (const ti of p.groupedTypes) {
      b.push(`## ${ti.name}\n\n`);
      writeDoc(b, ti.doc.summary);
      b.push(typeDeclBlock(ti), "\n");
    }
    return b.join("");
  }

  if (p.freeFunc) {
    renderSection(model, b, 2, p.freeFunc, "");
    return b.join("");
  }

  const ti = p.ti!;
  writeDoc(b, ti.doc.summary);
  renderExamples(b, ti.doc);

  if (ti.kind === "class") {
    if (!p.service || p.ti !== p.service) {
      const decl = classDeclBlock(ti);
      if (decl) b.push(decl, "\n");
    }
    // Constructor first (how you obtain the value), then static factories,
    // then service ("factory") methods, then methods on the value itself.
    if (ti.ctor) {
      renderSection(model, b, 2, ti.ctor, "");
    }
    for (const method of [...ti.staticMethods].sort((a, c) =>
      a.name.localeCompare(c.name),
    )) {
      renderSection(model, b, 2, method, "");
    }
    if (p.service) {
      renderServiceMethods(
        model,
        b,
        p.service,
        2,
        serviceAccessCaption(model, p.service.name),
      );
    }
    if (p.ti !== p.service) {
      renderObjectMethods(model, b, ti, 2);
    }
  } else {
    b.push(typeDeclBlock(ti), "\n");
  }

  // Dependent "companion" classes reached through a property on this class
  // (e.g. sandbox.filesystem) are documented inline as a namespace, one
  // heading level below this type's own methods.
  for (const c of p.companions) {
    renderCompanion(model, b, p.label, c);
  }

  return b.join("");
}

/**
 * Documents a companion class as a namespace section introduced by its
 * accessor path (e.g. "## Sandbox.filesystem"), with the companion's methods
 * nested one level deeper.
 */
function renderCompanion(
  model: PkgModel,
  b: string[],
  primary: string,
  c: Companion,
): void {
  b.push(`## ${primary}.${c.propName}\n\n`);
  const doc = c.propDoc.summary.trim() ? c.propDoc : c.ti.doc;
  writeDoc(b, doc.summary);
  if (c.isService) {
    renderServiceMethods(model, b, c.ti, 3, "");
  } else {
    renderObjectMethods(model, b, c.ti, 3);
  }
}

/**
 * Renders one section per service method, in declaration order. caption, when
 * non-empty, is shown under every method heading to name the call path (e.g.
 * via modal.volumes); it is empty for companion namespaces, where the section
 * heading already conveys the access path.
 */
function renderServiceMethods(
  model: PkgModel,
  b: string[],
  svc: TypeDeclInfo,
  depth: number,
  caption: string,
): void {
  for (const method of svc.methods) {
    renderSection(model, b, depth, method, caption);
  }
}

/**
 * Renders one section per public method declared on the class itself, sorted
 * alphabetically. These carry no call-path caption: they are plain methods on
 * the type the page documents. Only the merged service methods, which are
 * reached through a ModalClient property rather than the value itself, are
 * called out (see renderServiceMethods).
 */
function renderObjectMethods(
  model: PkgModel,
  b: string[],
  ti: TypeDeclInfo,
  depth: number,
): void {
  for (const method of [...ti.methods].sort((a, c) =>
    a.name.localeCompare(c.name),
  )) {
    renderSection(model, b, depth, method, "");
  }
}

/**
 * The caption shown under a merged service method, naming the ModalClient
 * property that exposes the service (e.g. modal.volumes). It is empty when the
 * service is not reached through ModalClient.
 */
function serviceAccessCaption(model: PkgModel, svcName: string): string {
  const prop = model.serviceAccessors.get(svcName);
  if (!prop) return "";
  return `_Accessed via \`modal.${prop}\`_`;
}

/**
 * Writes a heading (at the given depth), an optional caption naming how the
 * callable is reached, the signature code block, the doc comment, and the
 * folded XxxParams property list (if the final argument is one).
 */
function renderSection(
  model: PkgModel,
  b: string[],
  depth: number,
  method: MethodInfo,
  caption: string,
): void {
  b.push(`${"#".repeat(depth)} ${method.name}\n\n`);
  if (caption) b.push(caption, "\n\n");
  b.push(codeBlock(method.signature), "\n");

  if (method.doc.deprecated !== undefined) {
    const text = method.doc.deprecated || "This method is deprecated.";
    b.push(`**Deprecated.** ${oneLine(linkRefsToCode(text))}\n\n`);
  }
  writeDoc(b, method.doc.summary);

  // Positional-argument docs from @param tags. The trailing params-struct
  // argument is skipped: its type gets the full **Parameters** block below.
  const argDocs = method.doc.params.filter(
    (p) => !(method.paramsTypeName && p.name === method.lastParamName),
  );
  if (argDocs.length > 0) {
    for (const p of argDocs) {
      b.push(`- \`${p.name}\`: ${oneLine(linkRefsToCode(p.text))}\n`);
    }
    b.push("\n");
  }

  if (method.paramsTypeName) {
    renderParams(model, b, method.paramsTypeName);
  }

  if (method.doc.returns) {
    b.push(`**Returns:** ${oneLine(linkRefsToCode(method.doc.returns))}\n\n`);
  }
  if (method.doc.throws.length > 0) {
    b.push("**Raises:**\n\n");
    for (const t of method.doc.throws) {
      const desc = oneLine(linkRefsToCode(t.text));
      b.push(t.type ? `- \`${t.type}\`` : "-", desc ? `: ${desc}\n` : "\n");
    }
    b.push("\n");
  }
  renderExamples(b, method.doc);
}

function renderExamples(b: string[], doc: DocInfo): void {
  for (const example of doc.examples) {
    b.push(example.trim(), "\n\n");
  }
}

/**
 * Documents a method's XxxParams options type: the type name, its doc
 * comment, and one Markdown list item per property. The type is always shown
 * when a method takes one, even with no properties, so the reader sees which
 * options type the method accepts.
 */
function renderParams(model: PkgModel, b: string[], typeName: string): void {
  b.push(`**Parameters** (\`${typeName}\`)\n\n`);

  const ti = model.types.get(typeName);
  if (ti) writeDoc(b, ti.doc.summary);

  const lines: string[] = [];
  const members = resolveObjectMembers(model, typeName);
  for (const member of members) {
    if (!ts.isPropertySignature(member) || !member.name) continue;
    if (!ts.isIdentifier(member.name) && !ts.isStringLiteral(member.name)) {
      continue;
    }
    const name = member.name.text;
    if (!isPublicName(name)) continue;
    const doc = docFor(member, member.getSourceFile());
    if (doc.hidden) continue;
    const typeText = member.type
      ? oneLine(member.type.getText(member.getSourceFile()))
      : "unknown";
    // Optionality renders as the TypeScript `?` suffix, transcribing the
    // property signature; required fields are the unmarked ones.
    const optional = member.questionToken ? "?" : "";
    let line = `- \`${name}${optional}\` (\`${typeText}\`)`;
    const desc = oneLine(linkRefsToCode(doc.summary));
    if (desc) line += `: ${desc}`;
    lines.push(line);
  }

  if (lines.length > 0) {
    b.push(lines.join("\n"), "\n\n");
  } else {
    b.push("_No configurable options._\n\n");
  }
}

/**
 * Returns the property members of a params type, following at most a chain of
 * type-alias references (e.g. SidecarExecParams -> SandboxExecParams). Both
 * object-literal type aliases and interfaces are supported.
 */
function resolveObjectMembers(
  model: PkgModel,
  typeName: string,
): readonly ts.TypeElement[] {
  const seen = new Set<string>();
  let current: string | undefined = typeName;
  while (current && !seen.has(current)) {
    seen.add(current);
    const ti = model.types.get(current);
    if (!ti) return [];
    if (ti.kind === "interface") {
      return (ti.node as ts.InterfaceDeclaration).members;
    }
    if (ti.kind !== "typeAlias") return [];
    const target = (ti.node as ts.TypeAliasDeclaration).type;
    if (ts.isTypeLiteralNode(target)) return target.members;
    if (ts.isTypeReferenceNode(target) && ts.isIdentifier(target.typeName)) {
      current = target.typeName.text;
      continue;
    }
    return [];
  }
  return [];
}

/**
 * Renders a class's declaration block: its public fields and getters, keeping
 * each member's doc comment as an inline comment. Returns "" when the class
 * has no public data members.
 */
function classDeclBlock(ti: TypeDeclInfo): string {
  if (ti.properties.length === 0) return "";
  const lines = [`class ${ti.name} {`];
  for (const prop of ti.properties) {
    let line = "  " + prop.declLine;
    const comment = oneLine(linkRefsToCode(prop.doc.summary));
    if (comment) line += " // " + comment;
    lines.push(line);
  }
  lines.push("}");
  return codeBlock(lines.join("\n"));
}

/**
 * Renders a non-class type's declaration block by slicing its source text
 * (which keeps member doc comments in place for interfaces). For error
 * classes on the aggregate page, renders a one-line class heading showing the
 * inheritance chain instead.
 */
function typeDeclBlock(ti: TypeDeclInfo): string {
  if (ti.kind === "class") {
    let line = `class ${ti.name}`;
    if (ti.extendsName) line += ` extends ${ti.extendsName}`;
    return codeBlock(line);
  }
  const text = ti.node
    .getText(ti.sf)
    .replace(/^export\s+/, "")
    .replace(/^declare\s+/, "");
  return codeBlock(text);
}

// --- signature and name helpers ---

/**
 * Renders a declaration's signature by slicing the source text from the start
 * of the declaration (after any JSDoc) to the start of its body, dedenting
 * continuation lines by the declaration's own indentation.
 */
function signatureText(
  node:
    | ts.MethodDeclaration
    | ts.ConstructorDeclaration
    | ts.FunctionDeclaration,
  sf: ts.SourceFile,
): string {
  const start = node.getStart(sf);
  const end = node.body ? node.body.getStart(sf) : node.getEnd();
  let text = sf.text.slice(start, end).trimEnd();
  if (text.endsWith("{")) text = text.slice(0, -1).trimEnd();
  if (text.endsWith(";")) text = text.slice(0, -1);

  const lineStart = sf.text.lastIndexOf("\n", start) + 1;
  const indent = sf.text.slice(lineStart, start);
  if (/^[ \t]*$/.test(indent) && indent.length > 0) {
    text = text
      .split("\n")
      .map((line, i) =>
        i > 0 && line.startsWith(indent) ? line.slice(indent.length) : line,
      )
      .join("\n");
  }
  return text.replace(/^export\s+/, "");
}

/**
 * Returns the name of the final parameter's type when it references an
 * XxxParams type (directly, or as the base of an intersection like
 * `SandboxExecParams & { mode: "binary" }`), else undefined.
 */
function lastParamsTypeName(
  node: ts.SignatureDeclarationBase,
): string | undefined {
  const params = node.parameters;
  if (params.length === 0) return undefined;
  const last = params[params.length - 1];
  if (!last.type) return undefined;
  return paramsTypeNameFromNode(last.type);
}

function paramsTypeNameFromNode(node: ts.TypeNode): string | undefined {
  if (ts.isTypeReferenceNode(node) && ts.isIdentifier(node.typeName)) {
    return isParamsType(node.typeName.text) ? node.typeName.text : undefined;
  }
  if (ts.isIntersectionTypeNode(node)) {
    for (const part of node.types) {
      const name = paramsTypeNameFromNode(part);
      if (name) return name;
    }
  }
  return undefined;
}

function lastParamName(node: ts.SignatureDeclarationBase): string | undefined {
  const params = node.parameters;
  if (params.length === 0) return undefined;
  const name = params[params.length - 1].name;
  return ts.isIdentifier(name) ? name.text : undefined;
}

/**
 * Returns the bare identifier name of a type reference (generics stripped),
 * or undefined for anything else (unions, literals, qualified names).
 */
function baseTypeName(node: ts.TypeNode): string | undefined {
  if (ts.isTypeReferenceNode(node) && ts.isIdentifier(node.typeName)) {
    return node.typeName.text;
  }
  return undefined;
}

/** Returns the bare name of a class's extends-clause target, if any. */
function heritageName(cls: ts.ClassDeclaration): string | undefined {
  for (const clause of cls.heritageClauses ?? []) {
    if (clause.token !== ts.SyntaxKind.ExtendsKeyword) continue;
    const expr = clause.types[0]?.expression;
    if (expr && ts.isIdentifier(expr)) return expr.text;
  }
  return undefined;
}

// --- output helpers ---

function codeBlock(s: string): string {
  return "```typescript\n" + s.replace(/\n+$/, "") + "\n```\n";
}

/**
 * Writes a doc comment as a prose block (followed by a blank line), rewriting
 * TSDoc link tags into code spans. A no-op when the comment is empty.
 */
function writeDoc(b: string[], summary: string): void {
  const doc = linkRefsToCode(summary).trim();
  if (doc) b.push(doc, "\n\n");
}

/** Collapses a doc comment into a single trimmed line for inline use. */
function oneLine(s: string): string {
  return s.split(/\s+/).filter(Boolean).join(" ").trim();
}

// The JS SDK reference landing page, written to intro.md alongside the
// generated per-type pages. It is authored as a sibling intro.md and read at
// generation time. The Python, CLI, and Go references likewise emit their
// intro into the generated output, so the frontend treats every reference
// surface's landing page uniformly as generated Markdown (the empty URL id
// resolves to intro.md). The title is derived from the first heading; the
// description frontmatter feeds OpenGraph metadata.
function introDoc(): string {
  return fs.readFileSync(
    fileURLToPath(new URL("./intro.md", import.meta.url)),
    "utf8",
  );
}

export function writeOutput(
  outDir: string,
  model: PkgModel,
  pages: Page[],
  intro: string,
): void {
  fs.mkdirSync(outDir, { recursive: true });

  // The reference landing page, generated as intro.md alongside the per-type
  // pages; the empty URL id resolves to it.
  fs.writeFileSync(path.join(outDir, "intro.md"), intro);

  const items: { label: string; category: string }[] = [];

  // Pages are written flat (one file per type), mirroring the Python and Go
  // outputs; the type name is both the filename stem and the URL id segment.
  for (const p of pages) {
    const content = renderPage(model, p);
    fs.writeFileSync(path.join(outDir, `${p.label}.md`), content);
    items.push({ label: p.label, category: pageCategory(p) });
  }

  fs.writeFileSync(
    path.join(outDir, "sidebar.json"),
    JSON.stringify({ items }),
  );
}

function main(): void {
  const args = process.argv.slice(2);
  if (args.length < 1) {
    process.stderr.write(
      "usage: gen_reference_docs <output_dir> [<source_dir>]\n",
    );
    process.exit(2);
  }
  const outDir = args[0];
  const srcDir = args[1] ?? "src";

  const model = buildModelFromDir(srcDir);
  const pages = buildPages(model);
  writeOutput(outDir, model, pages, introDoc());
  process.stdout.write(
    `Wrote ${pages.length} JS reference pages to ${outDir}\n`,
  );
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  main();
}
