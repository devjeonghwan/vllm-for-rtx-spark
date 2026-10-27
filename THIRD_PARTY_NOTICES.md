# Third-party components

- vLLM 0.32.0: Apache-2.0. Selected Windows build and compatibility changes from SystemPanic/vllm-windows are incorporated in the patches. Builds start from upstream vLLM; the SystemPanic repository is not a separate install dependency. Revisions are recorded in `patches/vllm/manifest.json` and `sources.lock.json`.
- Qwen3.8 integration is based on myllmbox/qwen38-flash-next-recipe v5.3.1 and its pinned image. Original notices accompany the reused files.
- The CSF SM121a GPU object is the original object extracted from the recipe's `libmbx_marlin_csf.so`. It is bundled unchanged; its CUDA kernel source is not included. The Windows host code is included.
- b12x sources and policies retain their original Apache-2.0 notices. CuTe-generated kernels and NVIDIA DSL sources retain their separate notices; the compiler is a build dependency and is not bundled.
- The added kernel sources are ordinary files under `third_party/vllm-kernels/`, recovered from the pinned recipe image. Their original source notices are preserved.
- PyTorch, CUDA and NVIDIA FlashAttention are installed from their official distributions. They are not built or redistributed by this project.
- The optional draft HEAD2 basis is the original recipe data file, not a trained or modified model supplied by this project.
- Bundled Python libraries and native helpers retain their original license files. Model weights are obtained separately under the model's license.
- oss-harmony 0.0.11 (Apache-2.0) and llguidance 1.7.6 (MIT) include native ARM64 bindings and their original source under `third_party/`.
