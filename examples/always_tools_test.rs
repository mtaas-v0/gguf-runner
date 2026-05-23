use std::fs;
use gguf_runner::Tool;
use serde_json::Value;

struct ListUsersTool;
impl Tool for ListUsersTool {
    fn name(&self) -> &str { "list_users" }
    fn description(&self) -> &str { "List all users in the system. No arguments." }
    fn call(&mut self, _args: &Value) -> Result<Value, String> {
        Ok(serde_json::json!({"users": ["alice", "bob", "carol"]}))
    }
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "./Qwen3.5-2B-Q4_K_M.gguf".to_string());
    let data = fs::read(&path).expect("model file");
    eprintln!("loaded model: {path}");
    let data: &'static [u8] = Box::leak(data.into_boxed_slice());
    let mut rt = gguf_runner::EmbeddedRuntime::load_from_bytes(data).expect("load");
    rt.set_debug(true);

    let system = "You are a helpful admin assistant.";
    let mut history: Vec<(String, String)> = Vec::new();

    let turns = [
        "What is the capital of France?",       // simple chat, no tool
        "List all users",                       // should use list_users tool
        "Thanks. How many users were there?",   // follow-up
    ];

    for (i, prompt) in turns.iter().enumerate() {
        eprintln!("\n=== Turn {} ===\n> {prompt}", i + 1);
        let mut tool = ListUsersTool;
        let mut tools: Vec<&mut dyn Tool> = vec![&mut tool];
        let rx = rt.generate_with_tools(&history, prompt, system, &mut tools, 5).expect("generate");
        let mut response = String::new();
        for token in rx {
            eprint!("{token}");
            response.push_str(&token);
        }
        eprintln!();
        history.push((prompt.to_string(), response));
    }
}
