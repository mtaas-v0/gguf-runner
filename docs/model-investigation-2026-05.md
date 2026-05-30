# Multi-Turn and Tool-Calling Investigation (May 2026)

This document records a multi-day investigation into multi-turn coherence and
tool-calling support in `EmbeddedRuntime`. It's intended as a reference for
future debugging — when something looks wrong, start by re-reading what
behaviour we already characterised before instrumenting again.

## Context

`EmbeddedRuntime` is the library entry point: load a GGUF model from bytes,
then call `generate()`, `generate_collect()`, or `generate_with_tools()` over
a `(user, assistant)` history. The downstream consumer triggering this work
was [everlock](/Users/jens/tmp/everlock) (`crates/backend-ai-ssh/src/module.rs`),
which exposes the model through an SSH admin shell with optional tools.

Three failure modes drove the investigation:

1. CLI one-shot worked but REPL/embedded mode produced garbage on the same
   prompt.
2. Multi-turn conversations regressed turn-over-turn — the model regurgitated
   its previous answer instead of addressing the new question.
3. Tool calls only worked behind a keyword heuristic; meta-questions about
   tools ("what tools can you access?") fell through the heuristic and the
   model had no idea tools existed.

## Models tested

All models live next to the binary in `/Users/jens/tmp/gguf-runner/`:

| File | Arch | Quantization | Size on disk |
|---|---|---|---|
| `Bonsai-1.7B.gguf` (= `everlock/Bonsai-1.7B-Q1_0.gguf`) | qwen3 | Q1_0 (BIN1_41 in the codebase) | 248 MB |
| `Qwen3.5-0.8B-Q4_K_M.gguf` | qwen35 | Q4_K_M | 533 MB |
| `Qwen3.5-2B-Q4_K_M.gguf` | qwen35 | Q4_K_M | 1.3 GB |

Bonsai and the everlock model are byte-identical (`shasum -a 256` confirmed
`3d7c6c90dd98717a203adb22d5eacd2581850e40aa5327e144b97766cae5f7e3`).

## Test harness

Two example programs (`cargo build --release --example ...`):

- `examples/multiturn_test.rs` — loads a model via `EmbeddedRuntime`, runs
  a fixed sequence of four turns through `generate()`, prints each token
  to stderr. Optional first CLI arg overrides the model path.
- `examples/always_tools_test.rs` — defines a `ListUsersTool` returning
  `{"users": ["alice","bob","carol"]}`, runs three turns through
  `generate_with_tools()`. Optional first CLI arg overrides the model path.
- `examples/sysprompt_test.rs` — runs the same question with several system
  prompts of increasing complexity to isolate which wording confuses a model.

All three were used with the default `EmbeddedRuntime::load_from_bytes`
settings (greedy decoding `temperature = 0`, model's `top_k`/`top_p` from
`general.sampling.*` GGUF hints, `repeat_penalty = 1.1`, `repeat_last_n = 64`).
Stochastic sampling was tested via `set_temperature(0.7).set_top_k(40)...`.

To inspect what was actually fed to the model, `EmbeddedRuntime::set_debug(true)`
was added; it forwards `RuntimeEvent::Log` messages from the runtime to
stderr. The most useful per-step log is "Top 5 logits: ... | stop:
<|im_end|>(rank=N, logit=X)" emitted in the generation loop.

## Roadblocks and what they actually were

### 1. CLI looked fine, embedded looked broken — sampling defaults diverged

CLI `--prompt` produced "The capital of France is Paris." consistently.
Embedded mode regularly produced garbled output. They are the same code
path; the difference was the `GenerationSettings` they were initialised
with.

`load_with_debug_mode` (CLI) called `read_gguf_sampling_hints` and applied
`general.sampling.temp/top_k/top_p` from the GGUF. `load_from_bytes`
(embedded) used hardcoded `temperature=0.7, top_k=0, top_p=0.9`. With
`top_k=0` the sampler runs full-vocabulary multinomial over ~150k tokens;
on a noisy 1-bit model the `<|im_end|>` stop token (rank ~38 at the best
position, logit 8.96) is statistically dwarfed by content tokens (logit
13+). Hits were lottery — explained why every fresh REPL session was
worse than the last (different time-seeded RNG).

**Fix**: `load_from_bytes` now also calls `read_gguf_sampling_hints` and
defaults to **greedy decoding** for predictable embedded behaviour. Setters
(`set_temperature`, `set_top_k`, `set_top_p`, `set_repeat_penalty`,
`set_sampling_seed`) let callers opt back into stochastic sampling.

### 2. The chat template — Bonsai's empty `<think>` block

The single biggest correctness bug. The Bonsai chat template (in
`tokenizer.chat_template`) ends `add_generation_prompt` with:

```
<|im_start|>assistant
<think>

</think>


```

— i.e. an *empty* `<think>\n\n</think>\n\n` block already closed. The model
was trained to see this and produce the answer directly after it.
`vendors/qwen_common.rs` was injecting just `<think>\n` (opened, never
closed) and waiting for the model to generate think content. The model
had never seen that pattern.

**Fix part A**: `vendors/mod.rs` now sets `Config.qwen_chat_template_uses_empty_think
= true` when the template literal contains `<think>\n\n</think>` (either
literal newlines or escaped `\\n`). When that flag is set, the
runtime overrides `think_mode` to `ThinkMode::No` and the encoder emits
`<|im_start|>assistant\n<think>\n\n</think>\n\n`.

**Fix part B**: this also needs to apply to **past assistant messages** in
multi-turn conversations. The model generated turn N's answer against a
prompt ending with the empty think block, so when reconstructing turn N
in the history for turn N+1, that empty block must be present *between*
the assistant role tag and the content. Without this, the history the
model sees on turn N+1 doesn't match what it actually generated against,
and the model breaks.

The patch is in `vendors/qwen_common.rs` inside
`encode_qwen3_messages_with_think_style`:

```rust
if matches!(message.role, ChatRole::Assistant)
    && inject_forced_think_prompt
    && think_mode == ThinkMode::No
    && !message.content.contains("</think>")
{
    tokenizer.bpe_encode("<think>\n\n</think>\n\n", &mut temp);
    tokens.extend_from_slice(&temp);
}
```

This is why "Can you tell me about Beijing?" after asking about Paris
went from `"Okay, let me something about.ijing?"` to
`"The capital of China is Beijing."`.

### 3. Loop detection didn't fire on inline phrase repetition

The pre-existing guards `repeated_text_suffix_bytes` (needs a 64-byte
exact suffix match in two adjacent windows) and `repeated_long_line`
(needs 24+ char lines, plural, split on `\n`) were designed for
newline-separated output. Bonsai's typical loop is `"The capital of
france. The capital of france. The..."` — one long line, slight
variations like `"The The capital"` that break exact-byte matching.

**Fix**: added `repeated_inline_phrase(output)` in
`src/app/generation.rs`. It looks for an 8–80 byte substring appearing
3+ times non-overlapping in the last 1024 bytes of `output`, descending
from the longest candidate. Also expanded `repeated_cycle_period`'s
window list to `[4, 6, 8, 12, ...]` so tight 3–4-token cycles in the
hidden think phase get caught. Both checks now fire unconditionally
every 4 tokens past the first 8, not gated on `deterministic_loop_guard`
or `temperature == 0`.

### 4. `top_k=0` blocked the stop token *less* than `top_k=20` initially

The Bonsai GGUF has `general.sampling.top_k = 20` as a hint. Applying
that broke things until the empty-think fix went in: the stop token's
best rank was 38 with the broken prompt, so a top-20 filter excluded it.
After the chat-template fix, the stop token climbs to rank 1 (logit
15.18) immediately after "Paris." — well inside the top-20 window.
GGUF top_k hints are re-applied since the prompt structure now matches
what the model expects.

### 5. The 1-bit model can't handle ANY tool catalog in the system prompt

Three protocol variations were tried in `generate_with_tools` for the
Bonsai model:

1. Custom JSON-only protocol (`{"type":"tool_call",...}` / `{"type":"final"}`)
2. Qwen3 native `<tool_call>{"name":...}` / `<tool_response>...`
3. Bare-minimum "Tools: ..." listing with a one-line call syntax

All three produced complete garbage on Bonsai Q1_0 — even for "What is
2+2?" with the simplest tool description. Once the system prompt
contains anything more elaborate than "You are a helpful assistant.",
the 1-bit quantization noise drowns out instruction-following capacity.
No protocol change in the library can fix this; it's a model capability
problem.

### 6. Qwen3 vs Qwen3.5 tool-call format

The two model families use **different** tool-call syntaxes inside
`<tool_call>...</tool_call>`:

- Qwen3 (Bonsai parent): `{"name": "...", "arguments": {...}}` JSON object.
- Qwen3.5: an XML function-call block:
  `<function=name><parameter=k>v</parameter>...</function>`.

The library used to only parse the JSON form, so when Qwen3.5-0.8B emitted
its native XML the parser silently returned `None` and the visible output
was empty (we'd stripped the tool_call block). `extract_tool_call` now
tries (1) strict JSON, (2) XML function-call parse, (3) a loose
"`\"name\"\\s*:\\s*\"X\"`" extractor for cases where the small model emits
slightly broken JSON like `{"name":"list_users","arguments":{"}}`.

### 7. Greedy decoding on Qwen3.5 with stop-text literals

`vendors/qwen_common.rs`'s `QWEN_STOP_TEXT_LITERALS` includes
`</response>`, `</user>`, `</assistant>` etc. These were designed to
clean up structured-output runs. With the Everlock system prompt and
greedy decoding on Qwen3.5, the model's top first-token choice was
`</` followed by `response` followed by `>` — three tokens, stop-literal
match, empty visible output.

Lowercase ungrammatical prompts ("what is capital of france?") triggered
this path more reliably than well-formed ones ("What is the capital of
France?"). With stochastic sampling the model occasionally picks the
right alternative ("The"); with greedy it always picks `</`. The
work-around in practice is to use well-formed prompts (which everlock
SSH input typically is) or stochastic sampling; the stop literals are
still useful for other models so we didn't remove them.

## Tool-calling redesign

`generate_with_tools` originally enforced a custom JSON-only protocol
(`{"type":"tool_call",...}` / `{"type":"final"}`). The model had to
produce JSON for *every* response, which broke ordinary chat. The current
version uses Qwen3's native protocol:

1. System prompt is augmented with a `# Tools` block containing JSON
   specifications inside `<tools>...</tools>` XML tags — verbatim from
   what Qwen3 was trained on.
2. The model can either emit `<tool_call>...</tool_call>` or respond in
   plain prose.
3. Tool results are fed back wrapped in `<tool_response>...</tool_response>`
   as the next user turn.
4. When the model produces no `<tool_call>`, that response is treated as
   the final answer (after stripping any hallucinated `<tool_call>` or
   `<tool_response>` artifacts from the visible output).

Setters added to `EmbeddedRuntime`:

```rust
rt.set_temperature(0.7)
  .set_top_k(20)
  .set_top_p(0.85)
  .set_repeat_penalty(1.1)
  .set_sampling_seed(Some(42))
  .set_debug(true);
```

Strip helper `strip_tool_call_blocks` removes paired
`<tool_call>...</tool_call>` and `<tool_response>...</tool_response>`
blocks plus orphan closing tags (the model sometimes emits
`</tool_call>` without an opening tag).

## Results

The same Everlock-style four-turn conversation, run via
`./target/release/examples/multiturn_test <model.gguf>`:

| Turn | Bonsai Q1_0 | Qwen3.5-0.8B Q4_K_M | Qwen3.5-2B Q4_K_M |
|---|---|---|---|
| `What is the capital of France?` | "The capital of France is Paris." | "The capital of France is **Paris**." | "The capital of France is Paris." |
| `Do you know any Everlock tools?` | "The capital of france is paris." ✗ | "No, I don't have access to any Everlock tools." | "No, I don't know any specific tools..." |
| `What tools can you access?` | "user" ✗ | "I don't have access to any Everlock tools." | "As an AI, I don't have direct access..." |
| `Can you tell me about Beijing?` | loops on France ✗ | "Beijing is the capital of China. It is located in eastern China..." | bullet-pointed Beijing facts |

Tool calling via `./target/release/examples/always_tools_test <model.gguf>`
(`list_users` tool registered, always passed regardless of question):

| Test | Bonsai Q1_0 | Qwen3.5-0.8B Q4_K_M | Qwen3.5-2B Q4_K_M |
|---|---|---|---|
| Non-tool question | garbage tokens | clean direct answer ✓ | clean direct answer ✓ |
| Tool needed | garbage | calls tool but never converges to a final answer (`max_tool_calls` reached) | calls tool, produces final answer (with stripped artifacts) |
| Follow-up using cached result | n/a | n/a (hit `max_tool_calls`) | "There were 3 users: alice, bob, carol." ✓ |

Throughput on a single M-series Mac (release build, default settings):

| Model | tok/s |
|---|---|
| Bonsai Q1_0 | ~8 |
| Qwen3.5-0.8B Q4_K_M | ~45 |
| Qwen3.5-2B Q4_K_M | ~20 |

## Recommendations

- **Default model choice for everlock-style chat**: Qwen3.5-0.8B Q4_K_M.
  6× the throughput of Bonsai, multi-turn coherence intact, acknowledges
  knowledge gaps instead of regurgitating. Tools won't reliably complete
  end-to-end, but the existing heuristic in
  `everlock/crates/backend-ai-ssh/src/module.rs` covers that case.
- **If reliable tool calling is required**: Qwen3.5-2B Q4_K_M. Two-thirds
  the throughput of 0.8B but completes the agent loop and recalls tool
  results across turns.
- **Bonsai Q1_0 is not viable for anything beyond single-turn trivia.** Keep
  it around only as a regression test for the 1-bit code path.

## Files touched during the investigation

- `src/app/embed.rs` — `EmbeddedRuntime` API: `generate`, `generate_collect`,
  `generate_with_tools`, the `Tool` trait, setters, tool-call extractors.
- `src/app/generation.rs` — `read_gguf_sampling_hints`, embedded-runtime
  defaults, sampling/debug setters, `repeated_inline_phrase`, expanded
  cycle windows.
- `src/vendors/qwen_common.rs` — past-assistant empty-think block injection
  for `qwen3`/`qwen35`/`qwen3moe`/`qwen3vl` families.
- `src/vendors/mod.rs` — `qwen_chat_template_uses_empty_think` detection.
- `src/engine/types.rs` — `Config.qwen_chat_template_uses_empty_think` field.
- `src/lib.rs` — re-exports `EmbeddedRuntime` and `Tool`.
- `examples/multiturn_test.rs`, `examples/always_tools_test.rs`,
  `examples/sysprompt_test.rs` — repro harnesses.

## How to re-run the experiments

```bash
# Multi-turn coherence on a given model:
./target/release/examples/multiturn_test ./Qwen3.5-0.8B-Q4_K_M.gguf

# Tool-calling end-to-end:
./target/release/examples/always_tools_test ./Qwen3.5-2B-Q4_K_M.gguf

# System-prompt sensitivity sweep (Qwen3.5 hardcoded inside, edit if needed):
./target/release/examples/sysprompt_test

# CLI one-shot baseline (sampling hints + stop-token tracking visible with --debug):
./target/release/gguf-runner \
  --model ./Bonsai-1.7B.gguf \
  --prompt 'What is the capital of France?' \
  --debug --max-tokens 20 --seed 1
```

The default sampling for all examples is greedy
(`temperature = 0`) — deterministic, no seed needed. Switch to
stochastic via the setters at the top of each example file if reproducing
edge cases that depend on RNG.
