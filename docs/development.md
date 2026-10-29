# Development

Each project release supports one pinned vLLM revision. This release uses vLLM 0.32.0 at `5df63a7203cdb9328d824a27b682e77c7b9928be`. A later project release can update vLLM after integration and validation.

Change `PROJECT_VERSION` in `src/version.py` for a project release. Its wheel version is derived as `<vllm version>+rtx_spark.<project version>`.

## vLLM sources

`patches/vllm/manifest.json` specifies the upstream commit, build requirements, ordered patch filenames, and file/folder copy rules. `sources.lock.json` records third-party provenance. Each release has one vLLM revision.

```text
patches/vllm/             manifest.json and *.patch
src/runtime/vllm/         adapter
src/recipes/              model integration and helpers
src/kernels/              maintained kernel sources and providers
third_party/vllm-kernels/ external kernel sources
```

Patch filenames are relative to `patches/vllm/`. Copy sources are relative to the project repository root; destinations are relative to the assembled vLLM source tree. Files, folders, and empty files are supported; an unexpected existing destination fails.

`scripts/apply_source_changes.py` provides `apply_source_changes(source, manifest_path, source_root=...)`. Library build scripts call it with an explicit manifest after preparing an isolated Git source tree. It applies ordered patches and copy rules; fetching, compilation and build order remain with the caller. It does not discover other manifests.

Keep maintained code in `src/` by role. It is packaged under `vllm._rtx_spark` without separate copy rules. External source bundles belong in `third_party/`.

A maintained kernel compiled into a vLLM extension can also use a copy rule, for example `src/kernels/example/native` to `csrc/rtx_spark/example`. Add the corresponding CMake change as a patch; copying a source does not compile it.

For a vLLM update, port the patches and adapter to the new pinned revision and check the copied dependencies. Use a fresh prepared source directory after changes.

`scripts/export_vllm_changes.py` is a developer tool: it exports changes from an already ported source tree into a new staging directory, preserving repository-relative patch and copy-input paths. It does not port upstream automatically or export the maintained `src/` package. `scripts/build_native_assets.py` builds native support assets; `scripts/load_source_package.py` loads the checkout's private package for tools and tests.

## Models and architectures

Add a package under `src/recipes/` with `prepare(args, options)`, `validate(config, options)`, and an `OPTIONS` tuple. Optional `matches(args)` and `ARCHITECTURES` identify its recipe. Add its lazy module entry to `register_builtin_models()` in `src/recipes/__init__.py`.

For an architecture already supported by vLLM, use its existing implementation. A new implementation can expose `MODEL_CLASSES = {"ArchitectureName": "package.module:ModelClass"}`; the plugin registers these lazy paths with vLLM's ModelRegistry. Model implementations and helpers belong in their recipe package.

`register_model()` provides the same interface for registration from Python. Follow registration of new architecture classes with the idempotent plugin `register()` call. Keep recipe metadata CPU-only and model implementations lazy.

Use package-relative imports within helpers and fully qualified imports from vLLM. Recipe helper directories are not added to `sys.path`.

An explicit `rtx_spark.model` takes priority; multiple automatic matches fail. Preparation fixes the recipe for that engine, including no match, and final configuration validates the same choice. Automatically selected names are copied into the engine's options for workers. The vLLM hook passes its original EngineArgs to final validation; selection is not process-global.

Keep private selection state on EngineArgs. Do not add undeclared attributes to VllmConfig: MTP clones it with vLLM's strict config replacement utility.

## Optimization options

Declare a setting once in the recipe's `OPTIONS`. Reusable definitions belong in `src/optimizations/`; a recipe includes them when supported.

```python
from vllm._rtx_spark.optimizations import Option

def apply_setting(args, value):
    ...  # Connect to the supported implementation.

OPTIONS = (Option("setting", int, 7, apply=apply_setting),)
```

The runtime resolves defaults, validates types and optional `validate(args, value)` callbacks, prepares the recipe, and calls `apply(args, value)`. Final engine configuration is checked without reapplying callbacks.

```yaml
additional-config:
  rtx_spark:
    model: model-name
    optimizations:
      setting: 7
```

Flat option keys remain supported. Conflicting flat and nested values are rejected. Model-specific defaults stay in the recipe.

## Kernels and checks

CuTe and CSF providers remain under `src/kernels/`. Preserve tensor layouts, workspace, stream, and lifetime contracts when replacing them. Asset metadata retains filenames, ABI requirements and launch symbols; repeated file integrity hashes are not required.

A provider registration module exposes `register()`, which only registers CPU metadata through `cute.register_runtime()`, `csf.register_csf_provider()` or `cute.builder.register_provider()`. It must not load native libraries, model weights or execute kernels during registration.

List shipped modules in `kernels.providers.PROVIDER_MODULES`, or declare a packaged module before starting workers:

```python
from vllm._rtx_spark.kernels.providers import register_provider_module

register_provider_module("vllm._rtx_spark.kernels.example_provider")
```

The example requires an implemented module. The common startup path runs each declared module once per process and passes programmatic declarations to Windows spawn workers through the environment. Provider lookups and kernel builds use the same initialization. Direct registration of a callback in the parent alone does not make it available to workers.

Run `python -m unittest discover -s tests -p '*_test.py'` for CPU contracts. Check a fresh composed source tree after changing patches or copy rules. Compilation and GPU validation are separate steps.
