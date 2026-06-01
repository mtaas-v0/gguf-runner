// Items in this module are used by the binary crate. When the library crate is linted
// in isolation (cargo clippy without --bin) they appear unused because the lib only
// exports EmbeddedRuntime and does not re-export binary-only code.
#![allow(dead_code)]

use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};

use crate::app::events::{RuntimeEvent, RuntimeEventCallback};
use crate::app::generation::ModelRuntime;
use crate::vendors::{ChatMessage, ChatRole};

/// A tool that can be called by the model during generation.
///
/// Implement this trait and pass a slice of `&mut dyn Tool` to
/// [`EmbeddedRuntime::generate_with_tools`] to give the model access to
/// custom functionality (web search, database lookups, calculations, etc.).
///
/// The model communicates via a plain JSON protocol:
/// - To call a tool: `{"type":"tool_call","tool":"<name>","args":{...}}`
/// - To signal completion: `{"type":"final"}`
///
/// After the model signals completion a final natural-language response is
/// generated and streamed back to the caller.
pub trait Tool {
    /// Unique name the model uses to invoke this tool.
    fn name(&self) -> &str;

    /// Human-readable description shown to the model in the system prompt.
    /// Be specific: what the tool does, what arguments it expects.
    fn description(&self) -> &str;

    /// Execute the tool with the arguments the model provided.
    ///
    /// Return a JSON value that will be fed back to the model as the tool
    /// result. On error, return `Err` — the error message is also fed back
    /// so the model can adapt.
    fn call(&mut self, args: &serde_json::Value) -> Result<serde_json::Value, String>;
}

/// A loaded model ready for stateless multi-turn generation.
///
/// Callers manage conversation history as plain `(user, assistant)` string
/// pairs. `EmbeddedRuntime` handles chat-template encoding, context trimming,
/// and token streaming internally.
pub struct EmbeddedRuntime {
    inner: ModelRuntime,
}

impl EmbeddedRuntime {
    /// Load a model from bytes compiled into the binary (e.g. `include_bytes!`).
    ///
    /// Blocks until the model is fully loaded — weights are mapped and
    /// verified. Call once and reuse the returned instance.
    ///
    /// Defaults to **greedy decoding** (`temperature = 0.0`) for predictable,
    /// reproducible output. Use [`set_temperature`](Self::set_temperature) and
    /// the other setters below to opt into stochastic sampling.
    pub fn load_from_bytes(data: &'static [u8]) -> Result<Self, String> {
        Ok(Self {
            inner: ModelRuntime::load_from_bytes(data)?,
        })
    }

    /// Load a model from a filesystem path at runtime.
    ///
    /// Use this instead of [`load_from_bytes`](Self::load_from_bytes) when the
    /// model is too large to embed at compile time (e.g. multimodal vision
    /// models of 2 GB+).  The real path is preserved internally so that mmproj
    /// sidecar files are discovered automatically the first time an image is
    /// passed to [`generate_with_image`](Self::generate_with_image).
    pub fn load_from_file(path: &std::path::Path) -> Result<Self, String> {
        Ok(Self {
            inner: ModelRuntime::load_from_file(path, false)?,
        })
    }

    /// Load the mmproj vision projector from bytes embedded in the binary.
    ///
    /// Call this once after [`load_from_bytes`] to enable image inference without
    /// a sidecar file on disk.  The bytes must be a valid GGUF mmproj file
    /// (e.g. `mmproj-SmolVLM-256M-Instruct-f16.gguf`) passed via `include_bytes!`.
    pub fn load_mmproj_from_bytes(&mut self, data: &'static [u8]) -> Result<(), String> {
        self.inner.load_mmproj_from_bytes(data)
    }

    /// Generate text for an image + prompt pair, returning the complete output.
    ///
    /// The image must be a file on the local filesystem — pass the path as-is;
    /// gguf-runner reads and preprocesses the image internally.
    ///
    /// On the first call the mmproj vision projector sidecar is located next to
    /// the model file and loaded; subsequent calls reuse it.
    pub fn generate_with_image(
        &mut self,
        image_path: &std::path::Path,
        prompt: &str,
        system_prompt: &str,
    ) -> Result<String, String> {
        let image_str = image_path
            .to_str()
            .ok_or_else(|| "image path contains non-UTF8 characters".to_string())?;
        self.inner
            .generate_text_with_images(prompt, system_prompt, &[image_str.to_string()], false)
    }

    /// Set the sampling temperature.
    ///
    /// `0.0` (default) → greedy decoding: always picks the highest-probability
    /// token. Reproducible regardless of seed.
    ///
    /// `>0.0` → stochastic sampling. Higher values produce more diverse output
    /// but can degrade quality on heavily quantized models.
    pub fn set_temperature(&mut self, temperature: f32) -> &mut Self {
        self.inner.set_temperature(temperature);
        self
    }

    /// Restrict sampling to the top `k` highest-probability tokens.
    /// `0` disables the filter. Only takes effect when `temperature > 0`.
    pub fn set_top_k(&mut self, top_k: usize) -> &mut Self {
        self.inner.set_top_k(top_k);
        self
    }

    /// Nucleus sampling cutoff. Only takes effect when `top_k > 0`.
    pub fn set_top_p(&mut self, top_p: f32) -> &mut Self {
        self.inner.set_top_p(top_p);
        self
    }

    /// Penalty applied to tokens that appeared in the recent output.
    /// `1.0` (default) disables the penalty. Values like `1.1`–`1.2`
    /// discourage repetition.
    pub fn set_repeat_penalty(&mut self, repeat_penalty: f32) -> &mut Self {
        self.inner.set_repeat_penalty(repeat_penalty);
        self
    }

    /// Fixed RNG seed for reproducible stochastic sampling. `None` uses a
    /// time-based seed. Ignored when `temperature == 0.0`.
    pub fn set_sampling_seed(&mut self, seed: Option<u64>) -> &mut Self {
        self.inner.set_sampling_seed(seed);
        self
    }

    /// Build a RAG knowledge index from in-memory `(source_name, markdown_content)` pairs.
    ///
    /// `encoder_bytes` is a GGUF embedding model (e.g. `include_bytes!("nomic-embed.gguf")`).
    /// `docs` is a slice of `(source_label, markdown_text)` pairs; the source labels appear
    /// in the context block shown to the model so they should be human-readable paths.
    ///
    /// Every subsequent [`generate`](Self::generate) call will embed the user query,
    /// retrieve the most relevant chunks, and prepend them to the system prompt automatically.
    pub fn load_rag_from_embedded_docs(
        &mut self,
        encoder_bytes: &'static [u8],
        docs: &[(&str, &str)],
    ) -> Result<String, String> {
        self.inner.load_rag_from_embedded_docs(encoder_bytes, docs)
    }

    /// Load a precomputed RAG index that was serialized at build time.
    ///
    /// `encoder_bytes` is the GGUF embedding model used to embed the runtime
    /// query.  `index_bytes` is the output of [`build_serialized_rag_index`] —
    /// a flat byte buffer containing every chunk plus its precomputed embedding.
    ///
    /// Unlike [`load_rag_from_embedded_docs`](Self::load_rag_from_embedded_docs),
    /// this path does **no** per-chunk embedding work at startup, so the model
    /// is ready to serve immediately even for large doc sets.
    pub fn load_rag_from_serialized_bytes(
        &mut self,
        encoder_bytes: &'static [u8],
        index_bytes: &[u8],
    ) -> Result<String, String> {
        self.inner
            .load_rag_from_serialized_bytes(encoder_bytes, index_bytes)
    }

    /// Build a serialized RAG index from `(source, markdown)` pairs and return
    /// its bytes.  Intended to be called from a `build.rs` script so the
    /// resulting bytes can be embedded via `include_bytes!`.
    ///
    /// `encoder_bytes` is the GGUF embedding model used at build time; it must
    /// match (or share a vector space with) the encoder loaded at runtime via
    /// [`load_rag_from_serialized_bytes`].
    pub fn build_serialized_rag_index(
        encoder_bytes: &'static [u8],
        docs: &[(&str, &str)],
        max_chars_per_chunk: usize,
        max_tokens_per_chunk: usize,
    ) -> Result<Vec<u8>, String> {
        let mut encoder = crate::rag::encoder::DocumentEncoder::load_from_bytes(encoder_bytes)?;
        let index = crate::rag::RagIndex::build_from_text_slices(
            docs,
            &mut encoder,
            max_chars_per_chunk,
            max_tokens_per_chunk,
        )?;
        index.save_to_bytes()
    }

    /// Force hidden think mode regardless of what the model's chat template specifies.
    ///
    /// Some models (e.g. Qwen3.5) default to `ThinkMode::No` based on their chat template,
    /// which pre-fills an empty `<think></think>` block. When the model produces no visible
    /// output after that block, there is no retry. Calling this method switches to
    /// `ThinkMode::Hidden` instead, which enables an automatic retry with thinking suppressed
    /// whenever the first pass yields empty output.
    pub fn use_hidden_think_mode(&mut self) -> &mut Self {
        self.inner.set_think_mode_hidden();
        self
    }

    /// Enable verbose debug logging (prompt tokens, top-k logits per step,
    /// stop-token rank). Prints to stderr. Off by default.
    pub fn set_debug(&mut self, enabled: bool) -> &mut Self {
        self.inner.set_debug_mode(enabled);
        self
    }

    /// Generate a response and stream tokens back through an `mpsc` channel.
    ///
    /// Generation runs synchronously on the calling thread and blocks until
    /// complete. Each text fragment is sent to the channel as it is produced,
    /// and the sender is dropped when generation finishes — so iterating the
    /// returned `Receiver` to completion is all that is needed:
    ///
    /// ```no_run
    /// # let mut rt: gguf_runner::EmbeddedRuntime = todo!();
    /// let rx = rt.generate(&[], "What is 2+2?", "You are helpful")?;
    /// for token in rx {
    ///     print!("{token}");
    /// }
    /// # Ok::<_, String>(())
    /// ```
    ///
    /// `history` is a slice of prior `(user, assistant)` turns. Pass an empty
    /// slice for a single-turn exchange.
    pub fn generate(
        &mut self,
        history: &[(String, String)],
        input: &str,
        system_prompt: &str,
    ) -> Result<Receiver<String>, String> {
        let messages = build_messages(history, input);
        let (tx, rx) = channel::<String>();

        let cb: RuntimeEventCallback = Arc::new(move |event: RuntimeEvent| match event {
            RuntimeEvent::Output(text) => {
                let _ = tx.send(text);
            }
            RuntimeEvent::Log(log) => {
                eprintln!("[debug] {}", log.message);
            }
            _ => {}
        });

        self.inner.set_runtime_event_callback(Some(cb));
        let result = self
            .inner
            .generate_chat_messages_for_repl(&messages, system_prompt);
        // Dropping the stored Arc closes the Sender, so `rx` iteration will end.
        self.inner.set_runtime_event_callback(None);

        result?;
        Ok(rx)
    }

    /// Generate a response and return the complete text as a single `String`.
    ///
    /// Convenience wrapper around [`generate`](Self::generate) for callers
    /// that do not need per-token streaming.
    pub fn generate_collect(
        &mut self,
        history: &[(String, String)],
        input: &str,
        system_prompt: &str,
    ) -> Result<String, String> {
        let rx = self.generate(history, input, system_prompt)?;
        Ok(rx.into_iter().collect())
    }

    /// Generate a response with tool support.
    ///
    /// The model can call any of the provided `tools` zero or more times before
    /// producing a final answer. The call loop runs synchronously; the final
    /// natural-language response is streamed back through the returned
    /// `Receiver<String>` just like [`generate`](Self::generate).
    ///
    /// `max_tool_calls` caps the number of tool invocations per request to
    /// prevent runaway loops.
    ///
    /// # Example
    /// ```no_run
    /// use gguf_runner::Tool;
    /// use serde_json::Value;
    ///
    /// struct WeatherTool;
    /// impl Tool for WeatherTool {
    ///     fn name(&self) -> &str { "get_weather" }
    ///     fn description(&self) -> &str { "Get current weather. Args: {\"city\": string}" }
    ///     fn call(&mut self, args: &Value) -> Result<Value, String> {
    ///         let city = args["city"].as_str().unwrap_or("unknown");
    ///         Ok(serde_json::json!({"temperature": "22°C", "condition": "sunny", "city": city}))
    ///     }
    /// }
    ///
    /// # let mut rt: gguf_runner::EmbeddedRuntime = todo!();
    /// let rx = rt.generate_with_tools(
    ///     &[],
    ///     "What's the weather in Paris?",
    ///     "You are a helpful assistant.",
    ///     &mut [&mut WeatherTool],
    ///     5,
    /// )?;
    /// for token in rx { print!("{token}"); }
    /// # Ok::<_, String>(())
    /// ```
    pub fn generate_with_tools(
        &mut self,
        history: &[(String, String)],
        input: &str,
        system_prompt: &str,
        tools: &mut [&mut dyn Tool],
        max_tool_calls: usize,
    ) -> Result<Receiver<String>, String> {
        let tool_system_prompt = build_tool_system_prompt(system_prompt, tools);

        // Accumulate tool call/result turns as additional history.
        let mut extended_history: Vec<(String, String)> = history.to_vec();
        let mut current_input = input.to_string();
        let mut call_count = 0usize;

        loop {
            let response =
                self.generate_collect(&extended_history, &current_input, &tool_system_prompt)?;

            match extract_tool_call(&response) {
                Some(call) => {
                    if call_count >= max_tool_calls {
                        return Err(format!(
                            "max_tool_calls ({max_tool_calls}) reached before final response"
                        ));
                    }
                    call_count += 1;

                    let result = tools
                        .iter_mut()
                        .find(|t| t.name() == call.name)
                        .map(|t| t.call(&call.arguments))
                        .unwrap_or_else(|| Err(format!("unknown tool: {}", call.name)));

                    let result_json = match result {
                        Ok(v) => v,
                        Err(e) => serde_json::json!({"error": e}),
                    };

                    // Feed the result back to the model wrapped in <tool_response> tags
                    // (the Qwen3 native convention). The model then decides whether to
                    // call another tool or produce the final answer.
                    let tool_result_msg = format!(
                        "<tool_response>\n{}\n</tool_response>",
                        serde_json::to_string(&result_json).unwrap_or_else(|_| "{}".to_string())
                    );
                    extended_history.push((current_input.clone(), response.clone()));
                    current_input = tool_result_msg;
                }

                None => {
                    // No tool call — the response IS the final answer. Strip any
                    // protocol artifacts and stream it back through a channel.
                    let final_text = strip_tool_call_blocks(&response);
                    let (tx, rx) = channel::<String>();
                    if !final_text.is_empty() {
                        let _ = tx.send(final_text);
                    }
                    return Ok(rx);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tool protocol — Qwen3 native format
// ---------------------------------------------------------------------------
//
// Tools are described in the system prompt under a <tools>...</tools> block.
// The model calls a tool by emitting:
//   <tool_call>{"name": "<tool>", "arguments": {...}}</tool_call>
// and may interleave prose. Tool results are fed back wrapped in:
//   <tool_response>...</tool_response>
// If the model produces no <tool_call> tag, the response is treated as the
// final natural-language answer.

struct ToolCallRequest {
    name: String,
    arguments: serde_json::Value,
}

/// Find a `<tool_call>...</tool_call>` block in `text` and parse its body.
/// Returns `None` if no valid tool call is present.
///
/// Accepts three body formats (in order of preference):
///   1. Qwen3 / generic JSON:    `{"name": "...", "arguments": {...}}`
///   2. Qwen3.5 native XML:      `<function=name><parameter=k>v</parameter>...</function>`
///   3. Loose fallback: any `"name": "..."` substring with optional arguments object.
///
/// Tolerates the model emitting extra `<tool_call>` tags before the body
/// (some small models duplicate the opening tag).
fn extract_tool_call(text: &str) -> Option<ToolCallRequest> {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";

    // Locate the body between the LAST `<tool_call>` open and the FIRST
    // `</tool_call>` close after it. This handles `<tool_call><tool_call>...`.
    let close_idx = text.find(CLOSE)?;
    let before_close = &text[..close_idx];
    let open_idx = before_close.rfind(OPEN)?;
    let body = before_close[open_idx + OPEN.len()..].trim();

    // 1. Strict JSON
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(body)
        && let Some(name) = value.get("name").and_then(|v| v.as_str())
    {
        let arguments = value
            .get("arguments")
            .cloned()
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
        return Some(ToolCallRequest {
            name: name.to_string(),
            arguments,
        });
    }

    // 2. Qwen3.5 native XML
    if let Some(call) = parse_xml_function_call(body) {
        return Some(call);
    }

    // 3. Loose: find `"name": "X"` and treat arguments as missing.
    extract_loose_name(body).map(|name| ToolCallRequest {
        name,
        arguments: serde_json::Value::Object(serde_json::Map::new()),
    })
}

/// Find `"name"\s*:\s*"VALUE"` in `text` and return VALUE. Tolerates malformed
/// JSON around it so we can recover when a small model emits a broken object.
fn extract_loose_name(text: &str) -> Option<String> {
    let key_idx = text.find("\"name\"")?;
    let after_key = &text[key_idx + "\"name\"".len()..];
    let colon_idx = after_key.find(':')?;
    let after_colon = after_key[colon_idx + 1..].trim_start();
    let after_colon = after_colon.strip_prefix('"')?;
    let end_idx = after_colon.find('"')?;
    Some(after_colon[..end_idx].to_string())
}

/// Parse the Qwen3.5 native function call XML:
///   `<function=NAME><parameter=K1>V1</parameter><parameter=K2>V2</parameter></function>`
fn parse_xml_function_call(body: &str) -> Option<ToolCallRequest> {
    let fn_open_marker = "<function=";
    let fn_open_idx = body.find(fn_open_marker)?;
    let after_fn = &body[fn_open_idx + fn_open_marker.len()..];
    let name_end = after_fn.find('>')?;
    let name = after_fn[..name_end].trim().to_string();
    let fn_body = &after_fn[name_end + 1..];
    let fn_body_end = fn_body.find("</function>").unwrap_or(fn_body.len());
    let fn_body = &fn_body[..fn_body_end];

    let mut args = serde_json::Map::new();
    let param_open = "<parameter=";
    let param_close = "</parameter>";
    let mut remaining = fn_body;
    while let Some(p_start) = remaining.find(param_open) {
        let after = &remaining[p_start + param_open.len()..];
        let Some(key_end) = after.find('>') else {
            break;
        };
        let key = after[..key_end].trim().to_string();
        let value_part = &after[key_end + 1..];
        let Some(value_end) = value_part.find(param_close) else {
            break;
        };
        let raw_value = value_part[..value_end].trim();
        // Try to parse as JSON (number, bool, object, array); otherwise treat as string.
        let value = serde_json::from_str::<serde_json::Value>(raw_value)
            .unwrap_or_else(|_| serde_json::Value::String(raw_value.to_string()));
        args.insert(key, value);
        remaining = &value_part[value_end + param_close.len()..];
    }
    Some(ToolCallRequest {
        name,
        arguments: serde_json::Value::Object(args),
    })
}

/// Remove any `<tool_call>...</tool_call>` and hallucinated
/// `<tool_response>...</tool_response>` blocks from `text` so the final answer
/// doesn't leak protocol artifacts to the caller.
fn strip_tool_call_blocks(text: &str) -> String {
    let stripped = strip_xml_block(text, "tool_call");
    let stripped = strip_xml_block(&stripped, "tool_response");
    stripped.trim().to_string()
}

fn strip_xml_block(text: &str, tag: &str) -> String {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut out = String::with_capacity(text.len());
    let mut remaining = text;
    while let Some(open_idx) = remaining.find(&open) {
        out.push_str(&remaining[..open_idx]);
        match remaining[open_idx + open.len()..].find(&close) {
            Some(close_rel) => {
                remaining = &remaining[open_idx + open.len() + close_rel + close.len()..];
            }
            None => {
                // Unterminated tag — drop the rest of the text after the open tag.
                remaining = "";
                break;
            }
        }
    }
    out.push_str(remaining);
    // Also remove orphan closing tags (the model sometimes emits </tool_call>
    // without a matching opening tag, especially on tool-trained models that
    // mix the protocol with prose).
    out.replace(&close, "")
}

fn build_tool_system_prompt(base: &str, tools: &[&mut dyn Tool]) -> String {
    // Use the Qwen3 official `# Tools` block with <tools><tool>...</tool></tools>
    // JSON schemas, the format the model was trained on. The instructions tell
    // the model to use tools only when they're actually relevant to the query,
    // and to write a plain prose answer otherwise — without this, the model
    // tends to call tools eagerly for questions that don't need them.
    let tool_specs: Vec<String> = tools
        .iter()
        .map(|t| {
            let spec = serde_json::json!({
                "name": t.name(),
                "description": t.description(),
            });
            serde_json::to_string(&spec).unwrap_or_default()
        })
        .collect();

    format!(
        "{base}\n\n\
# Tools\n\n\
You have access to the following functions, defined as JSON specs inside <tools></tools>:\n\
<tools>\n{}\n</tools>\n\n\
When a question needs information that one of the above functions provides, call it by emitting:\n\
<tool_call>\n{{\"name\": \"<function-name>\", \"arguments\": <args-json-object>}}\n</tool_call>\n\n\
The user (system) will reply with the result wrapped in <tool_response>...</tool_response>. Write your final answer to the user in plain prose using that result. If the question can be answered from general knowledge without any tool, just answer directly.",
        tool_specs.join("\n")
    )
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn build_messages(history: &[(String, String)], input: &str) -> Vec<ChatMessage> {
    let mut messages: Vec<ChatMessage> = history
        .iter()
        .flat_map(|(user, assistant)| {
            [
                ChatMessage {
                    role: ChatRole::User,
                    content: user.clone(),
                },
                ChatMessage {
                    role: ChatRole::Assistant,
                    content: assistant.clone(),
                },
            ]
        })
        .collect();
    messages.push(ChatMessage {
        role: ChatRole::User,
        content: input.to_string(),
    });
    messages
}
