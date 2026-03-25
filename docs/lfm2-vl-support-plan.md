# LFM2-VL Support Plan

This document is a handoff plan for adding support for LiquidAI LFM2-VL models such as:

- `LiquidAI_LFM2-VL-450M-GGUF_LFM2-VL-450M-Q4_0.gguf`
- `LiquidAI_LFM2-VL-450M-GGUF_mmproj-LFM2-VL-450M-Q8_0.gguf`

It captures:

- the current gap between `gguf-runner` and LFM2-VL
- the implementation phases
- the expected file touch points
- validation work
- reference material already inspected locally and online

## Current Conclusion

LFM2-VL is not a small metadata alias on top of an existing family.

Supporting it requires two new pieces:

1. A new `lfm2` text/runtime family.
2. A new `lfm2` multimodal backend for the `mmproj` sidecar and image prompt/injection path.

The current runner fails before inference starts because it sees `general.architecture = lfm2`, does not recognize it, falls back to llama-like defaults, and then mismatches tensor shapes.

## What We Verified

### Local text GGUF observations

Local file inspected:

- `LiquidAI_LFM2-VL-450M-GGUF_LFM2-VL-450M-Q4_0.gguf`

Observed metadata and structure:

- `general.architecture = lfm2`
- `general.type = image-text-to-text`
- model-specific keys include:
  - `lfm2.block_count`
  - `lfm2.context_length`
  - `lfm2.embedding_length`
  - `lfm2.feed_forward_length`
  - `lfm2.attention.head_count`
  - `lfm2.attention.head_count_kv`
  - `lfm2.rope.freq_base`
  - `lfm2.attention.layer_norm_rms_epsilon`
  - `lfm2.vocab_size`
  - `lfm2.shortconv.l_cache`
- vocab/special tokens include:
  - ChatML tokens: `<|im_start|>`, `<|im_end|>`
  - image sentinel: `<image>`
  - image boundary tokens: `<|image_start|>`, `<|image_end|>`
  - tile marker tokens: `<|img_row_1_col_1|>` ... `<|img_row_10_col_10|>`
  - thumbnail token: `<|img_thumbnail|>`
- tensor layout shows mixed recurrent and attention layers:
  - recurrent-style layers have `blk.N.shortconv.conv.weight`, `shortconv.in_proj.weight`, `shortconv.out_proj.weight`
  - attention-style layers have `blk.N.attn_q.weight`, `attn_k.weight`, `attn_v.weight`, `attn_output.weight`
  - q/k norm tensors exist on attention layers: `attn_q_norm.weight`, `attn_k_norm.weight`
- final norm tensor is `token_embd_norm.weight`, not `output_norm.weight`

### Local sidecar GGUF observations

Local file inspected:

- `LiquidAI_LFM2-VL-450M-GGUF_mmproj-LFM2-VL-450M-Q8_0.gguf`

Observed metadata and structure:

- `general.architecture = clip`
- `clip.projector_type = lfm2`
- metadata includes:
  - `clip.has_vision_encoder`
  - `clip.vision.projection_dim`
  - `clip.vision.image_size`
  - `clip.vision.patch_size`
  - `clip.vision.embedding_length`
  - `clip.vision.feed_forward_length`
  - `clip.vision.block_count`
  - `clip.vision.attention.head_count`
  - `clip.vision.image_mean`
  - `clip.vision.image_std`
  - `clip.vision.attention.layer_norm_epsilon`
  - `clip.vision.projector.scale_factor`
  - `clip.use_gelu`
- tensor groups include:
  - vision tower: `v.patch_embd.*`, `v.position_embd.weight`, `v.blk.*`, `v.post_ln.*`
  - projector/input norm: `mm.input_norm.*`, `mm.1.*`, `mm.2.*`

This does not match the runner's currently supported external vision families:

- Gemma3: `clip.projector_type = gemma3`
- Qwen3-VL / Qwen3.5: `clip.projector_type = qwen3vl_merger`

### Current runner failure

Observed failure against the local text GGUF:

- the runner logs `Model architecture: lfm2`
- the config falls back to generic defaults
- weight loading then fails on `token_embd.weight` shape mismatch

This confirms the text path must be taught how to parse `lfm2.*` metadata before anything else can work.

## Upstream Reference Material Already Parsed

The vendored `llama.cpp` tree already contains both text and multimodal support for LFM2/LFM2-VL.

### Text architecture references

Relevant files:

- `llama.cpp/src/models/lfm2.cpp`
- `llama.cpp/src/llama-model.cpp`
- `llama.cpp/src/llama-hparams.cpp`
- `llama.cpp/src/llama-arch.cpp`

Key upstream findings:

- `llama.cpp` recognizes `lfm2` and `lfm2moe` as distinct architectures.
- LFM2 is a hybrid stack:
  - recurrent layers are identified when `n_head_kv(layer) == 0`
  - attention layers are identified when `n_head_kv(layer) > 0`
- recurrent layers use a short-convolution block:
  - `shortconv.in_proj`
  - `shortconv.conv`
  - `shortconv.out_proj`
- attention layers use:
  - `wq`, `wk`, `wv`, `wo`
  - q/k RMS norm before RoPE
- recurrent state size depends on `shortconv.l_cache`:
  - upstream computes recurrent memory from `n_embd * (n_shortconv_l_cache - 1)`
- final norm uses the tensor name alias `token_embd_norm`

### Multimodal references

Relevant files:

- `llama.cpp/tools/mtmd/clip.cpp`
- `llama.cpp/tools/mtmd/mtmd.cpp`
- `llama.cpp/tools/mtmd/models/siglip.cpp`
- `llama.cpp/convert_hf_to_gguf.py`

Key upstream findings:

- LFM2-VL sidecars use `PROJECTOR_TYPE_LFM2`.
- The vision tower path is implemented on top of the SigLIP graph, not the Qwen3-VL graph.
- The projector path is:
  - patch merge permute / pixel unshuffle
  - optional input norm
  - 2-layer MLP using GELU
- `mtmd` prompt formatting for LFM2 uses:
  - begin token: `<|image_start|>`
  - end token: `<|image_end|>`
  - single-tile images: one image payload between begin/end
  - multi-tile images: row/column tile marker tokens like `<|img_row_%d_col_%d|>`
  - overview/thumbnail marker token `<|img_thumbnail|>`
- preprocessing for LFM2 supports:
  - variable-resolution images
  - smart resize with alignment to `patch_size * downsample_factor`
  - tiling for larger images
  - tile size `512`
  - candidate grids between 2 and 10 tiles

### Online reference material

These links were already checked:

- Hugging Face model docs:
  - https://huggingface.co/docs/transformers/en/model_doc/lfm2_vl
- Hugging Face model page:
  - https://huggingface.co/LiquidAI/LFM2-VL-450M-GGUF
- Hugging Face model tree:
  - https://huggingface.co/models?other=base_model%3Aquantized%3ALiquidAI%2FLFM2-VL-450M

Relevant extracted facts from the Transformers docs:

- LFM2-VL consists of:
  - an LFM2 language backbone
  - a SigLIP2 NaFlex vision encoder
  - a 2-layer MLP connector with pixel unshuffle
- larger images are split into non-overlapping `512x512` patches
- special tokens identify patch positions and thumbnail placement
- LFM2-VL-450M uses the LFM2-350M backbone

## Current `gguf-runner` Gaps

### 1. Architecture detection

Current code in `src/vendors/mod.rs` does not recognize `lfm2`.

Impact:

- config fields are populated with incorrect defaults
- multimodal backend remains `None`
- weight loading fails immediately

### 2. Config model is too generic for LFM2

Current `Config` in `src/engine/types.rs` does not capture:

- per-layer recurrent vs attention layout
- shortconv cache length
- LFM2-specific family flags

Current `Config.n_kv_heads` is a single scalar, but LFM2 needs per-layer semantics because recurrent layers effectively have `kv_heads = 0`.

### 3. Weight loader only supports existing families

Current `src/engine/weights.rs` can load:

- standard transformer layers
- BERT fused-QKV
- Qwen3Next SSM-style recurrent layers

It cannot load LFM2 shortconv layers.

### 4. Runtime only knows transformer and Qwen3Next recurrent paths

Current `src/engine/runtime/inference.rs` does not implement:

- the LFM2 shortconv recurrent block
- per-layer dispatch between shortconv and standard attention based on LFM2 layer metadata

### 5. Multimodal backend detection is hardcoded

Current `MultimodalBackend` only has:

- `None`
- `Gemma3`
- `Qwen3Vl`
- `Qwen35`

LFM2 needs its own backend.

### 6. Sidecar validation is too narrow

Current `validate_mmproj_for_backend(...)` only accepts:

- `clip.projector_type = gemma3`
- `clip.projector_type = qwen3vl_merger`

LFM2 needs:

- `clip.projector_type = lfm2`

### 7. Vision encoder path is missing

Current `src/engine/multimodal/mod.rs` only builds:

- `Gemma3VisionEncoder`
- `Qwen3VlVisionEncoder`

LFM2 needs a dedicated encoder implementation.

### 8. Image preprocessing is too simple

Current preprocessing in `src/engine/vision/preprocess.rs` covers fixed resize/crop policies but not:

- smart variable-resolution resize
- LFM2 grid selection
- tile slicing
- thumbnail/overview handling for multi-tile input

### 9. Prompt encoding and image injection are too specialized

Current multimodal request encoding assumes either:

- Gemma3 image placeholder pair
- Qwen vision wrapper tokens

Current image embedding injection in `src/engine/multimodal/injection.rs` assumes:

- one begin token
- repeated placeholder token
- one end token

That model does not naturally represent LFM2's tiled layout with row/column marker tokens and optional overview image sections.

## Recommended Implementation Phases

## Phase 1: Text-only `lfm2` bootstrap

Goal:

- load and run text-only LFM2 models correctly before touching multimodal execution

Work:

- add `ModelFamily::Lfm2`
- add `is_lfm2` to `Config`
- parse `lfm2.*` metadata in `build_config_from_gguf(...)`
- add config support for:
  - `shortconv_l_cache`
  - per-layer recurrent flags
  - per-layer KV head counts if needed
- add LFM2 vendor module for:
  - chat prompt encoding
  - request encoding
  - tokenizer policy
  - decode policy
- load `token_embd_norm.weight` as final norm for LFM2
- extend weight loading with shortconv tensors
- add runtime shortconv execution path

Success criteria:

- text-only LFM2 prompt runs end-to-end
- no tensor shape mismatches during init
- generation returns coherent output

## Phase 2: Structural cleanup for mixed-layer families

Goal:

- make the runtime and config model robust enough for LFM2 without special-case hacks everywhere

Work:

- introduce a generic way to express per-layer mode:
  - attention
  - recurrent-shortconv
- move family-specific layer selection into vendor/config setup rather than ad hoc runtime branching
- keep `app/` generic and preserve the existing architecture boundaries in `AGENTS.md`

Success criteria:

- LFM2 layer-mode decisions happen through config/runtime data, not hardcoded checks in orchestration code

## Phase 3: Sidecar detection and backend plumbing

Goal:

- recognize LFM2-VL as multimodal-capable and initialize its sidecar correctly

Work:

- extend `MultimodalBackend` with `Lfm2`
- update capability probing for LFM2 special tokens and tensor groups
- update mmproj file scoring hints in vendor policy
- allow `clip.projector_type = lfm2` in sidecar validation
- add clearer debug output for LFM2 sidecar pairing

Success criteria:

- LFM2 text GGUF reports `backend=lfm2`
- local sidecar auto-discovery finds the matching `mmproj`
- sidecar validation passes

## Phase 4: LFM2 vision encoder

Goal:

- execute the sidecar image encoder and projector natively inside `gguf-runner`

Work:

- add `src/engine/multimodal/lfm2.rs`
- add `VisionEncoder::Lfm2`
- parse LFM2 sidecar metadata:
  - patch size
  - embedding length
  - block count
  - head count
  - layer norm epsilon
  - image mean/std
  - projector scale factor
- implement the SigLIP-style vision forward path used by LFM2
- implement patch merge permute / pixel unshuffle
- implement optional input norm
- implement 2-layer projector MLP with GELU

Success criteria:

- image tensors can be encoded into text-dimension embeddings
- embedding dimension matches the text model embedding size

## Phase 5: LFM2 preprocessing and tiling

Goal:

- reproduce LFM2 image preparation semantics closely enough for correct prompting

Work:

- extend `src/engine/vision/preprocess.rs` with an LFM2-specific path
- support:
  - smart resize preserving aspect ratio
  - alignment to `patch_size * downsample_factor`
  - tile slicing for large images
  - overview/thumbnail handling if needed by this checkpoint family
- carry tile-grid metadata forward so prompt construction and embedding insertion can agree

Success criteria:

- preprocessed image batches match the intended LFM2 tile layout
- both single-image and tiled-image paths are supported

## Phase 6: Prompt construction and embedding injection

Goal:

- build the exact token structure expected by LFM2-VL around image embeddings

Work:

- add `src/vendors/lfm2.rs` request encoding support for:
  - ChatML prompt structure
  - `<image>` sentinel handling
  - begin/end image markers
  - tile marker tokens
  - thumbnail marker token if applicable
- generalize `src/engine/multimodal/injection.rs` so image injection can handle:
  - more than one placeholder token type
  - tile marker tokens that should remain literal
  - layout-specific embedding insertion

Success criteria:

- generated prompt token stream matches LFM2-VL expectations for single-tile and multi-tile images

## Phase 7: Validation, docs, and follow-up hardening

Goal:

- verify correctness and leave support maintainable

Work:

- add regression notes and docs updates
- run required repo validation commands
- test:
  - text-only generation
  - image captioning on a known test image
  - single-tile input
  - multi-tile input
- compare behavior with `llama.cpp` where practical

Success criteria:

- code passes repo checks
- model loads and runs in both text and image-text mode

## Expected File Touch Points

### Vendor/config layer

- `src/vendors/mod.rs`
- `src/vendors/lfm2.rs` (new)

Expected changes:

- family detection
- `lfm2.*` config parsing
- multimodal policy
- prompt/request encoding
- decode/tokenizer/runtime debug policy

### Engine types/runtime layer

- `src/engine/types.rs`
- `src/engine/weights.rs`
- `src/engine/runtime/inference.rs`
- `src/engine/kernels/math.rs` if a reusable shortconv helper is needed

Expected changes:

- LFM2 config/runtime state
- shortconv weights
- recurrent state allocation
- shortconv block execution

### Multimodal layer

- `src/engine/multimodal/mod.rs`
- `src/engine/multimodal/lfm2.rs` (new)
- `src/engine/multimodal/injection.rs`
- `src/engine/vision/preprocess.rs`

Expected changes:

- sidecar vision encoder
- preprocessing and tiling
- LFM2-specific embedding injection

### App/docs layer

- `src/app/generation.rs`
- `docs/module-structure.md` when architecture-related code lands
- `docs/features.md` when support becomes user-facing
- `docs/downloading-models.md` if LFM2 sidecar pairing needs documentation

Expected changes:

- sidecar detection and probe messaging
- updated documentation once implementation starts landing

## Design Guidance

Keep the existing repo boundaries from `AGENTS.md`:

- CLI/env additions stay in `src/cli.rs`
- orchestration stays in `src/app/`
- inference/runtime stays in `src/engine/`
- family-specific prompt/multimodal policy stays in `src/vendors/`

Specific guidance:

- do not branch on `config.is_lfm2` in `src/app/*` unless strictly unavoidable
- route LFM2-specific prompt/media behavior through vendor policies and vendor request encoders
- keep the runtime generic where possible by expressing LFM2 layer behavior in config/runtime data

## Open Questions To Resolve During Implementation

- Does the local 450M checkpoint always use a thumbnail/overview image in multi-tile mode, or is that only used on larger variants?
- Do we need per-layer `n_kv_heads` in `Config`, or is a boolean recurrent-mask plus standard head dimensions sufficient?
- Should LFM2 prompt encoding reuse current ChatML/Qwen helpers, or should it get its own dedicated implementation?
- How closely do we need to mirror upstream smart-resize heuristics before outputs become acceptable?
- Is it worth generalizing image injection now for tile layouts, or should LFM2 temporarily own a separate injection path?

## Suggested Work Order For Resuming Later

1. Implement text-only `lfm2` config detection and loading.
2. Make one small text-only LFM2 prompt run.
3. Add `MultimodalBackend::Lfm2` and sidecar validation.
4. Implement the LFM2 sidecar encoder.
5. Implement LFM2 preprocessing and tiled prompt layout.
6. Integrate image embedding injection.
7. Validate against the local 450M GGUF pair and compare behavior with `llama.cpp`.

## Validation Checklist For The Eventual Implementation

Run from repo root:

1. `cargo fmt --all --check`
2. `cargo clippy --all-targets --all-features`
3. `cargo check`
4. `rg -n "^use crate::\\*;" src/engine -g'*.rs'`
5. `rg -n "^use crate::engine::[A-Za-z0-9_:]+::\\*;" src/engine -g'*.rs'`
6. `rg -n "crate::cli::" src/engine -g'*.rs'`

If `cargo fmt --all --check` fails:

1. `cargo fmt --all`
2. rerun the checks above

## Quick Resume Summary

If picking this up later, start by remembering:

- the model is `lfm2`, not llama/qwen/gemma
- the text stack is hybrid shortconv + attention
- the final norm tensor is `token_embd_norm.weight`
- the sidecar is `clip.projector_type = lfm2`
- the image path is SigLIP-based with pixel unshuffle and a 2-layer MLP projector
- prompt formatting uses ChatML plus LFM2 image/tile tokens rather than the current Qwen or Gemma placeholder schemes
