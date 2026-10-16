# Architecture

Synchronicity 2 separates wrapper generation from the runtime used by generated wrappers.
Implementation modules register pure-async classes and functions at build time. Generated public
modules call the small runtime synchronizer without importing the code-generation package.

## Code-Generation Pipeline

The code generator has four layers:

1. `codegen.parsing` inspects live Python declarations and annotations. It is the only layer that
   resolves concrete Python types, source expressions, and Synchronicity 1 compatibility metadata.
2. `codegen.ir` contains frozen, data-only records. The records preserve parsed structure and
   wrapper intent without retaining live implementation objects or source-generation behavior.
3. `codegen.emission` turns IR into Python source. `TypeCodegen` implementations define public
   annotations and wrapper/implementation boundary expressions for each annotation IR shape.
4. `codegen.pipeline` orchestrates module parsing and emission. The CLI delegates to this layer.

```text
live declarations -> parsing -> IR -> emission -> generated Python modules
```

The IR boundary makes parser failures distinguishable from source-generation failures and lets the
two stages be tested independently. Unit tests mirror the package split under `test/unit/parsing`
and `test/unit/emission`.

## Naming

- `WrappedClassIR`, `WrappedFunctionIR`, and related declaration records describe wrapper intent.
  They do not represent runtime wrapper objects.
- Fields that contain IR use an `_ir` or `_irs` suffix, such as `annotation_ir`, `item_ir`, and
  `arm_irs`.
- `ObjectReferenceIR` identifies any implementation or wrapper object by module and qualified
  name. The containing field, such as `impl_ref` or `wrapper_ref`, communicates its role.
- `TypeCodegen` names emission behavior. The older `transformer` and `materialize` terminology is
  intentionally avoided because it obscured the parsing, representation, and emission boundaries.

## Import Policy

IR records do not calculate imports. `codegen.emission.imports` traverses IR and determines which
modules emitted source requires. This keeps output policy out of the data model.

## Runtime Boundary

Generated modules may import `synchronicity2` runtime modules or a vendored runtime package. They
must not import `synchronicity2.codegen`. The runtime owns the synchronizer thread, event loop, and
wrapper instance translation; all decisions about which types require translation are made during
code generation.
