# Implementing Tools for `EmbeddedRuntime`

`EmbeddedRuntime` supports tool-calling via the `Tool` trait. The model can
invoke your tools zero or more times before producing a final answer, and the
whole interaction is driven by a synchronous loop inside `generate_with_tools`.

> For background on why the protocol looks the way it does, what models we
> tested, and what works end-to-end, see
> [`model-investigation-2026-05.md`](./model-investigation-2026-05.md).

---

## How it works

When you call `generate_with_tools` the runtime:

1. Builds a system prompt that lists your tools as JSON specs inside a
   `<tools>...</tools>` block — the format Qwen3 / Qwen3.5 were trained on.
2. Runs the model and collects its full output for that turn.
3. If the output contains `<tool_call>...</tool_call>`, the matching tool's
   `call` method is invoked and the result is fed back to the model wrapped
   in `<tool_response>...</tool_response>`.
4. Steps 2–3 repeat until the model writes a plain-prose response with no
   tool call, or `max_tool_calls` is reached.
5. The final response is streamed back through the returned
   `Receiver<String>`.

Two tool-call body formats are accepted inside `<tool_call>...</tool_call>`:

- **JSON (Qwen3 / generic):** `{"name": "tool_name", "arguments": {...}}`
- **XML (Qwen3.5 native):** `<function=tool_name><parameter=k>v</parameter>...</function>`

A loose extractor also recovers tool names from slightly malformed JSON
(e.g. `{"name":"X","arguments":{"}}`) so small models that mangle the
syntax can still call tools.

---

## Implementing the `Tool` trait

```rust
use gguf_runner::Tool;
use serde_json::Value;

struct Calculator;

impl Tool for Calculator {
    fn name(&self) -> &str {
        "calculate"
    }

    fn description(&self) -> &str {
        // Be specific: the model reads this to know when and how to call the tool.
        "Evaluate a mathematical expression and return the numeric result. \
         Args: {\"expression\": \"<math expression as string>\"}"
    }

    fn call(&mut self, args: &Value) -> Result<Value, String> {
        let expr = args["expression"]
            .as_str()
            .ok_or("missing 'expression' argument")?;

        let result = evaluate(expr)?;

        Ok(serde_json::json!({ "result": result }))
    }
}
```

### Naming

- Use `snake_case` names (`get_weather`, not `GetWeather`).
- Keep names short and descriptive — the model includes the name in every call.

### Description

The description is the model's only documentation for your tool. Include:
- What the tool does
- The expected argument shape as a JSON snippet
- Any important constraints or return format

### Return value

Return a `serde_json::Value` that the model can read. A flat object with
clearly named fields works best:

```rust
// Good
Ok(serde_json::json!({ "temperature": "18°C", "condition": "cloudy" }))

// Less useful — the model has to guess what "42" means
Ok(serde_json::json!(42))
```

On error, return `Err(message)`. The runtime wraps it as
`{"error": "<message>"}` and feeds it back so the model can retry or adapt.

---

## Calling `generate_with_tools`

```rust
use gguf_runner::EmbeddedRuntime;

static MODEL: &[u8] = include_bytes!("../models/my-model.gguf");

fn main() -> Result<(), String> {
    let mut runtime = EmbeddedRuntime::load_from_bytes(MODEL)?;

    let mut calc = Calculator;
    let mut weather = WeatherTool;

    let rx = runtime.generate_with_tools(
        &[],                          // conversation history
        "What is 17 * 6, and what's the weather in Berlin?",
        "You are a helpful assistant.",
        &mut [&mut calc, &mut weather],
        10,                           // max tool calls per turn
    )?;

    for token in rx {
        print!("{token}");
    }
    println!();
    Ok(())
}
```

### Parameters

| Parameter | Description |
|---|---|
| `history` | Prior `(user, assistant)` turns. Pass `&[]` for the first message. |
| `input` | The user's current message. |
| `system_prompt` | Your base instructions for the model. Tool protocol rules are appended automatically. |
| `tools` | Mutable slice of `&mut dyn Tool`. Each tool can carry its own state. |
| `max_tool_calls` | Hard cap on tool invocations per request. Prevents runaway loops. |

---

## Sampling defaults and overrides

`EmbeddedRuntime::load_from_bytes` defaults to **greedy decoding**
(`temperature = 0.0`). Greedy is deterministic across runs, isn't seed
dependent, and avoids the worst pathologies of stochastic sampling on
heavily quantized models (the highest-probability token always wins, so
the stop token doesn't get lost in noise).

If you want diversity, opt in after loading:

```rust
let mut rt = EmbeddedRuntime::load_from_bytes(MODEL)?;
rt.set_temperature(0.7)
  .set_top_k(20)
  .set_top_p(0.85)
  .set_repeat_penalty(1.1)
  .set_sampling_seed(Some(42)); // reproducible stochastic runs
```

Other GGUF-supplied hints (`general.sampling.top_k`, `general.sampling.top_p`)
are read automatically and applied alongside the greedy temperature.

Turn on the runtime's debug logging (prompt tokens, top-k logits, stop-token
rank per step) with `rt.set_debug(true)` — `RuntimeEvent::Log` messages are
forwarded to stderr.

---

## Stateful tools

Because `call` takes `&mut self`, your tool can hold and update state across
multiple invocations within a single request:

```rust
struct SearchTool {
    client: HttpClient,
    calls_made: usize,
}

impl Tool for SearchTool {
    fn name(&self) -> &str { "web_search" }
    fn description(&self) -> &str {
        "Search the web. Args: {\"query\": \"<search terms>\"}"
    }
    fn call(&mut self, args: &serde_json::Value) -> Result<serde_json::Value, String> {
        self.calls_made += 1;
        let query = args["query"].as_str().ok_or("missing query")?;
        let results = self.client.search(query)?;
        Ok(serde_json::json!({ "results": results }))
    }
}
```

---

## Multi-turn conversations with tools

Pass prior turns in `history` to maintain context across requests:

```rust
let mut history: Vec<(String, String)> = Vec::new();

// First turn
let rx = runtime.generate_with_tools(&history, "What's 9 * 8?", system, tools, 5)?;
let response: String = rx.into_iter().collect();
history.push(("What's 9 * 8?".to_string(), response));

// Second turn — model remembers the previous exchange
let rx = runtime.generate_with_tools(&history, "Add 3 to that.", system, tools, 5)?;
for token in rx { print!("{token}"); }
```

Multi-turn coherence on Qwen3-style models relies on the encoder wrapping
past assistant messages with the same empty `<think>\n\n</think>\n\n`
block the model originally generated against. That's handled internally
in `vendors/qwen_common.rs` — you don't need to do anything.

---

## Without tools

For requests that don't need tools, use the simpler methods:

```rust
// Streaming — Receiver<String> yields tokens as they arrive
let rx = runtime.generate(&history, "Tell me a joke.", system)?;
for token in rx { print!("{token}"); }

// Blocking — returns the full response at once
let text = runtime.generate_collect(&history, "Summarise this text.", system)?;
println!("{text}");
```

---

## Model selection notes

Tool calling is much more demanding of the underlying model than plain
chat — the model has to follow a multi-step protocol over several
generation passes. From the May 2026 investigation:

| Model | Plain chat | Tool calling |
|---|---|---|
| Bonsai-1.7B Q1_0 (248 MB) | broken on anything multi-turn | not viable |
| Qwen3.5-0.8B Q4_K_M (533 MB) | excellent, ~45 tok/s | starts the loop but doesn't converge to a final answer |
| Qwen3.5-2B Q4_K_M (1.3 GB) | excellent, ~20 tok/s | full agent loop completes |

If `generate_with_tools` is in your hot path, pick at least the 2 GB-class
model. The 0.8 GB-class is great for chat-only.
