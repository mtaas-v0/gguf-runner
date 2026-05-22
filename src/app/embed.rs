// Items in this module are used by the binary crate. When the library crate is linted
// in isolation (cargo clippy without --bin) they appear unused because the lib only
// exports EmbeddedRuntime and does not re-export binary-only code.
#![allow(dead_code)]

use std::sync::Arc;

use crate::app::events::{RuntimeEvent, RuntimeEventCallback};
use crate::app::generation::ModelRuntime;
use crate::vendors::{ChatMessage, ChatRole};

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
    pub fn load_from_bytes(data: &'static [u8]) -> Result<Self, String> {
        Ok(Self {
            inner: ModelRuntime::load_from_bytes(data)?,
        })
    }

    /// Generate a response to `input`, calling `on_token` for each output
    /// token as it is produced.
    ///
    /// `history` is a slice of prior `(user, assistant)` turns for this
    /// session. Pass an empty slice for a single-turn exchange. The caller is
    /// responsible for appending the returned response to their history store.
    ///
    /// Returns the complete assistant response as a `String`.
    pub fn chat(
        &mut self,
        history: &[(String, String)],
        input: &str,
        system_prompt: &str,
        on_token: impl Fn(&str) + Send + Sync + 'static,
    ) -> Result<String, String> {
        // Build the full message list from history + current turn.
        let mut messages: Vec<ChatMessage> = history
            .iter()
            .flat_map(|(user, assistant)| {
                [
                    ChatMessage { role: ChatRole::User,      content: user.clone() },
                    ChatMessage { role: ChatRole::Assistant, content: assistant.clone() },
                ]
            })
            .collect();
        messages.push(ChatMessage { role: ChatRole::User, content: input.to_string() });

        // Accumulate the response while also calling the streaming callback.
        // Use Arc<Mutex> so the closure can own a clone and we can recover the
        // value after the callback is dropped.
        let response = Arc::new(std::sync::Mutex::new(String::new()));
        let response_cb = Arc::clone(&response);

        let cb: RuntimeEventCallback = Arc::new(move |event: RuntimeEvent| {
            if let RuntimeEvent::Output(text) = &event {
                on_token(text);
                if let Ok(mut buf) = response_cb.lock() {
                    buf.push_str(text);
                }
            }
        });

        self.inner.set_runtime_event_callback(Some(cb));
        let result = self.inner.generate_chat_messages_for_repl(&messages, system_prompt);
        self.inner.set_runtime_event_callback(None);

        result?;
        let text = Arc::try_unwrap(response)
            .map(|m| m.into_inner().unwrap_or_default())
            .unwrap_or_default();
        Ok(text)
    }
}
