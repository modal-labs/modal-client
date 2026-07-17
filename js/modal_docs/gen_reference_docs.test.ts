import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterAll, describe, expect, test } from "vitest";

import {
  buildModel,
  buildPages,
  linkRefsToCode,
  pageLabel,
  parseDocComment,
  renderPage,
  writeOutput,
  type Page,
  type PkgModel,
} from "./gen_reference_docs";

// A miniature SDK source set exercising the doc generator's key behaviors: a
// service class merged onto its entity page, XxxParams types folded into
// their methods (including an alias chain and an empty params type), a
// companion class reached via a getter, error classes grouped by their
// heritage chain, hidden members excluded, and the Function_ page naming.
const fixtureFiles: Record<string, string> = {
  "index.ts": `
export { ModalClient } from "./client";
export {
  Volume,
  VolumeService,
  VolumeFilesystem,
  type VolumeFromNameParams,
  type VolumeReloadParams,
  type VolumeDropParams,
  type VolumeListFilesParams,
  type VolumeCopyParams,
  type StdioBehavior,
} from "./volume";
export { Function_, FunctionService } from "./function";
export { NotFoundError, InternalFailure, VolumeSubError } from "./errors";
`,
  "client.ts": `
import { VolumeService } from "./volume";
import { FunctionService } from "./function";

/** The main client. */
export class ModalClient {
  readonly volumes: VolumeService;
  readonly functions: FunctionService;
  /** @ignore */
  readonly cpClient: unknown;
  constructor() {
    this.volumes = new VolumeService(this);
    this.functions = new FunctionService(this);
    this.cpClient = null;
  }
}
`,
  "volume.ts": `
import { type ModalClient } from "./client";

/** Optional parameters for {@link VolumeService#fromName client.volumes.fromName()}. */
export type VolumeFromNameParams = {
  /** Environment to look in. */
  environment?: string;
  /** Create the Volume if it is missing. */
  createIfMissing?: boolean;
  /** Region to search. */
  region: string;
};

/** Options for reload. */
export type VolumeReloadParams = {
  /** Force a reload. */
  force?: boolean;
};

/** VolumeDropParams are options for drop. */
export type VolumeDropParams = {};

/** Options for listFiles. */
export type VolumeListFilesParams = {
  /** Recurse into subdirectories. */
  recursive?: boolean;
};

/** Options for copy, aliasing the listFiles options. */
export type VolumeCopyParams = VolumeListFilesParams;

/** Controls stream behavior. */
export type StdioBehavior = "pipe" | "ignore";

/** Service for managing Volumes. */
export class VolumeService {
  readonly #client: ModalClient;
  constructor(client: ModalClient) {
    this.#client = client;
  }

  /** Reference a {@link Volume} by its name. See [multi word] brackets. */
  async fromName(
    name: string,
    params?: VolumeFromNameParams,
  ): Promise<Volume> {
    return new Volume(name);
  }
}

/** VolumeFilesystem operates on a Volume's files. */
export class VolumeFilesystem {
  /** List files in the Volume. See {@link Volume#reload Volume.reload()}. */
  async listFiles(params?: VolumeListFilesParams): Promise<string[]> {
    return [];
  }

  /** Copy files around. */
  async copy(params?: VolumeCopyParams): Promise<void> {}
}

/** Volume represents a Modal Volume. */
export class Volume {
  readonly volumeId: string;
  readonly #internal: number = 0;
  /**
   * @hidden
   * @internal
   */
  readonly _mountOptions: string = "";

  /** @ignore */
  constructor(volumeId: string) {
    this.volumeId = volumeId;
  }

  /** Filesystem provides filesystem operations for this Volume. */
  get filesystem(): VolumeFilesystem {
    return new VolumeFilesystem();
  }

  /** Reload the Volume metadata. */
  async reload(params?: VolumeReloadParams): Promise<void> {}

  /**
   * Drop the Volume.
   * @throws {NotFoundError} the Volume does not exist.
   * @throws {@link InternalFailure} the deletion fails transiently.
   */
  async drop(params?: VolumeDropParams): Promise<void> {}

  async readInto(): Promise<Uint8Array>;
  async readInto(binary: true): Promise<Uint8Array>;
  /** Read the Volume contents. */
  async readInto(binary?: boolean): Promise<Uint8Array> {
    return new Uint8Array();
  }

  /** @hidden */
  async hiddenMethod(): Promise<void> {}
}
`,
  "function.ts": `
import { type ModalClient } from "./client";

/** Service for managing Functions. */
export class FunctionService {
  readonly #client: ModalClient;
  constructor(client: ModalClient) {
    this.#client = client;
  }

  /** Reference a {@link Function_ Function} by its name. */
  async fromName(appName: string, name: string): Promise<Function_> {
    return new Function_();
  }
}

/** Represents a deployed Modal Function. */
export class Function_ {
  /** Call the Function remotely. */
  async remote(args: unknown[]): Promise<unknown> {
    return null;
  }
}
`,
  "errors.ts": `
/** Some resource was not found. */
export class NotFoundError extends Error {}

/** Does not end in "Error" but is still an error type. */
export class InternalFailure extends Error {}

/** Extends another error class rather than Error directly. */
export class VolumeSubError extends NotFoundError {}
`,
};

function buildFixturePages(): { model: PkgModel; byLabel: Map<string, Page> } {
  const model = buildModel(fixtureFiles);
  const byLabel = new Map(buildPages(model).map((p) => [p.label, p]));
  return { model, byLabel };
}

describe("buildPages", () => {
  test("merges services and folds params and companions", () => {
    const { model, byLabel } = buildFixturePages();

    // The service class is merged into the entity page; neither it nor any
    // XxxParams type gets a page of its own.
    expect(byLabel.has("Volume")).toBe(true);
    expect(byLabel.has("VolumeService")).toBe(false);
    expect(byLabel.has("VolumeFromNameParams")).toBe(false);
    expect(byLabel.has("VolumeReloadParams")).toBe(false);

    const volume = renderPage(model, byLabel.get("Volume")!);

    // Merged service method with its access-path caption (a sentence
    // fragment, no trailing period). Methods on the type itself get none.
    expect(volume).toContain("## fromName");
    expect(volume).toContain("Reference a `Volume` by its name.");
    expect(volume).toContain("_Accessed via `modal.volumes`_");
    // Folded params with property docs. Optional properties carry the
    // TypeScript `?` suffix; required ones are unmarked.
    expect(volume).toContain("**Parameters** (`VolumeFromNameParams`)");
    expect(volume).toContain(
      "- `environment?` (`string`): Environment to look in.",
    );
    expect(volume).toContain("- `region` (`string`): Region to search.");
    // Object methods are on the same page.
    expect(volume).toContain("## reload");
    // An empty params type is still documented: the type, its doc, and a note.
    expect(volume).toContain("## drop");
    expect(volume).toContain("**Parameters** (`VolumeDropParams`)");
    expect(volume).toContain("VolumeDropParams are options for drop.");
    expect(volume).toContain("_No configurable options._");
    // @throws tags render as a Raises list, with the bare type name extracted
    // from both the JSDoc ({Type}) and TSDoc ({@link Type}) forms.
    expect(volume).toContain("**Raises:**");
    expect(volume).toContain("- `NotFoundError`: the Volume does not exist.");
    expect(volume).toContain(
      "- `InternalFailure`: the deletion fails transiently.",
    );
    expect(volume).not.toContain("@link");

    // A companion class reached via a getter is folded in as a namespace
    // section (with its methods nested deeper), not given a page of its own.
    expect(byLabel.has("VolumeFilesystem")).toBe(false);
    expect(volume).toContain("## Volume.filesystem");
    expect(volume).toContain(
      "Filesystem provides filesystem operations for this Volume.",
    );
    expect(volume).toContain("### listFiles");
    // A params type aliasing another resolves through the chain.
    expect(volume).toContain("**Parameters** (`VolumeCopyParams`)");
    expect(volume).toContain(
      "`recursive?` (`boolean`): Recurse into subdirectories.",
    );
    // TSDoc links in prose become Markdown code spans; the raw link syntax
    // (which the docs site does not render) does not survive.
    expect(volume).toContain("See `Volume.reload()`.");
    expect(volume).not.toContain("{@link");
    // Multi-word bracketed text is left alone (it is not a TSDoc link).
    expect(volume).toContain("See [multi word] brackets.");

    // Only public members appear: no #-private or _-prefixed fields, no
    // @hidden methods, and no @ignore'd constructor section.
    expect(volume).toContain("volumeId");
    expect(volume).not.toContain("#internal");
    expect(volume).not.toContain("_mountOptions");
    expect(volume).not.toContain("hiddenMethod");
    expect(volume).not.toContain("new Volume(");
  });

  test("shows overload signatures, not the implementation", () => {
    const { model, byLabel } = buildFixturePages();
    const volume = renderPage(model, byLabel.get("Volume")!);

    expect(volume).toContain("## readInto");
    expect(volume).toContain("async readInto(): Promise<Uint8Array>");
    expect(volume).toContain(
      "async readInto(binary: true): Promise<Uint8Array>",
    );
    expect(volume).not.toContain("binary?: boolean");
    // The doc comes from the (only) documented declaration.
    expect(volume).toContain("Read the Volume contents.");
  });

  test("names the Function_ page Function", () => {
    const { model, byLabel } = buildFixturePages();

    expect(byLabel.has("Function")).toBe(true);
    expect(byLabel.has("Function_")).toBe(false);

    const fn = renderPage(model, byLabel.get("Function")!);
    expect(fn).toContain("# Function\n");
    // The service still matches its entity through the underscore suffix.
    expect(fn).toContain("## fromName");
    expect(fn).toContain("_Accessed via `modal.functions`_");
    expect(fn).toContain("## remote");
  });

  test("groups error classes onto one Errors page", () => {
    const { model, byLabel } = buildFixturePages();

    // Error classes are collected onto a single Errors page, not one page
    // each, detected by the heritage chain reaching Error (so
    // InternalFailure, which does not end in "Error", and VolumeSubError,
    // which extends another error class, are both grouped).
    expect(byLabel.has("Errors")).toBe(true);
    expect(byLabel.has("NotFoundError")).toBe(false);
    expect(byLabel.has("InternalFailure")).toBe(false);
    expect(byLabel.has("VolumeSubError")).toBe(false);

    const errs = renderPage(model, byLabel.get("Errors")!);
    expect(errs).toContain("## NotFoundError");
    expect(errs).toContain("Some resource was not found.");
    expect(errs).toContain("## InternalFailure");
    expect(errs).toContain("class VolumeSubError extends NotFoundError");
  });

  test("renders string-union type aliases as their declaration", () => {
    const { model, byLabel } = buildFixturePages();
    const out = renderPage(model, byLabel.get("StdioBehavior")!);
    expect(out).toContain('type StdioBehavior = "pipe" | "ignore"');
    expect(out).toContain("Controls stream behavior.");
  });

  test("excludes ModalClient internals but keeps its services listing", () => {
    const { model, byLabel } = buildFixturePages();
    const client = renderPage(model, byLabel.get("ModalClient")!);
    expect(client).toContain("readonly volumes: VolumeService;");
    expect(client).not.toContain("cpClient");
  });
});

describe("writeOutput", () => {
  const outDir = fs.mkdtempSync(path.join(os.tmpdir(), "jsdoc-test-"));
  afterAll(() => fs.rmSync(outDir, { recursive: true, force: true }));

  test("writes flat pages with intro and sidebar alongside", () => {
    const model = buildModel(fixtureFiles);
    writeOutput(outDir, model, buildPages(model), "# JS SDK Reference\n");

    // Pages are written flat into the output dir, with the landing page and
    // sidebar index alongside them. The service class is merged, so it has
    // no page, and Function_ is written under its display label.
    expect(fs.existsSync(path.join(outDir, "Volume.md"))).toBe(true);
    expect(fs.existsSync(path.join(outDir, "Function.md"))).toBe(true);
    expect(fs.existsSync(path.join(outDir, "intro.md"))).toBe(true);
    expect(fs.existsSync(path.join(outDir, "VolumeService.md"))).toBe(false);

    const sidebar = JSON.parse(
      fs.readFileSync(path.join(outDir, "sidebar.json"), "utf8"),
    ) as { items: { label: string; category: string }[] };
    const labels = sidebar.items.map((item) => item.label);
    expect(labels).toContain("Volume");
    expect(labels).toContain("Function");
    expect(labels).toContain("Errors");
    expect(labels).not.toContain("VolumeFilesystem");
    expect(sidebar.items.every((item) => item.category === "type")).toBe(true);
  });
});

describe("linkRefsToCode", () => {
  test.each([
    ["see {@link Sandbox}", "see `Sandbox`"],
    ["see {@link Sandbox}es", "see `Sandbox`es"],
    ["{@link Function_ Function}s", "`Function`s"],
    [
      "use {@link SandboxService#create client.sandboxes.create()}",
      "use `client.sandboxes.create()`",
    ],
    ["bare {@link Sandbox#exec}", "bare `Sandbox.exec`"],
    ["{@linkcode Volume}", "`Volume`"],
    ["{@link A} and {@link b.C}", "`A` and `b.C`"],
    // Non-link bracket text is left alone.
    ["[Go blog]: https://go.dev", "[Go blog]: https://go.dev"],
  ])("%s", (input, want) => {
    expect(linkRefsToCode(input)).toBe(want);
  });
});

describe("parseDocComment", () => {
  test("splits summary and block tags", () => {
    const doc = parseDocComment(`/**
 * Copy a local file into the Sandbox.
 *
 * More detail here.
 * @param localPath - Path on the local machine.
 * @param remotePath - Absolute path in the Sandbox.
 *   Continued on a second line.
 * @returns Nothing useful.
 * @throws {SandboxFilesystemError} the command fails.
 * @example
 * \`\`\`typescript
 * await sb.filesystem.copyFromLocal("a", "/b");
 * \`\`\`
 */`);
    expect(doc.summary).toBe(
      "Copy a local file into the Sandbox.\n\nMore detail here.",
    );
    expect(doc.params).toEqual([
      { name: "localPath", text: "Path on the local machine." },
      {
        name: "remotePath",
        text: "Absolute path in the Sandbox.\n  Continued on a second line.",
      },
    ]);
    expect(doc.returns).toBe("Nothing useful.");
    expect(doc.throws).toEqual([
      { type: "SandboxFilesystemError", text: "the command fails." },
    ]);
    expect(doc.examples).toHaveLength(1);
    expect(doc.examples[0]).toContain("copyFromLocal");
    expect(doc.hidden).toBe(false);
  });

  test("extracts the raised type from both @throws forms", () => {
    // JSDoc form: `@throws {Type} desc`.
    expect(
      parseDocComment("/** @throws {TimeoutError} too slow. */").throws,
    ).toEqual([{ type: "TimeoutError", text: "too slow." }]);
    // TSDoc form: `@throws {@link Type} desc`, including the code/label and
    // `#` member-path variants.
    expect(
      parseDocComment("/** @throws {@link TimeoutError} too slow. */").throws,
    ).toEqual([{ type: "TimeoutError", text: "too slow." }]);
    expect(
      parseDocComment("/** @throws {@linkcode TimeoutError} too slow. */")
        .throws,
    ).toEqual([{ type: "TimeoutError", text: "too slow." }]);
    expect(
      parseDocComment("/** @throws {@link Function_ Function} bad call. */")
        .throws,
    ).toEqual([{ type: "Function", text: "bad call." }]);
    expect(
      parseDocComment("/** @throws {@link Errors#TimeoutError} too slow. */")
        .throws,
    ).toEqual([{ type: "Errors.TimeoutError", text: "too slow." }]);
    // Braceless prose falls through with no type.
    expect(parseDocComment("/** @throws on bad input. */").throws).toEqual([
      { type: "", text: "on bad input." },
    ]);
  });

  test("flags hidden tags", () => {
    expect(parseDocComment("/** @hidden */").hidden).toBe(true);
    expect(parseDocComment("/** @ignore */").hidden).toBe(true);
    expect(parseDocComment("/** text\n * @internal\n */").hidden).toBe(true);
    expect(parseDocComment("/** plain */").hidden).toBe(false);
  });
});

describe("pageLabel", () => {
  test("strips the underscore suffix used to avoid JS keywords", () => {
    expect(pageLabel("Function_")).toBe("Function");
    expect(pageLabel("Sandbox")).toBe("Sandbox");
  });
});
