use std::fs;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "./Qwen3.5-2B-Q4_K_M.gguf".to_string());
    let data = fs::read(&path).expect("model file");
    let data: &'static [u8] = Box::leak(data.into_boxed_slice());
    eprintln!("loaded model: {path}");

    let mut rt = gguf_runner::EmbeddedRuntime::load_from_bytes(data).expect("load");
    let system = "You are Everlock AI. Answer clearly and concisely for an operator using the Everlock admin shell.";
    let mut history: Vec<(String, String)> = Vec::new();

    let turns = [
        "What is the capital of France?",
        "Do you know any Everlock tools?",
        "What tools can you access?",
        "Can you tell me something about Beijing?",
    ];

    for (i, prompt) in turns.iter().enumerate() {
        eprintln!("\n=== Turn {} ===\n> {prompt}", i + 1);
        let rx = rt.generate(&history, prompt, system).expect("generate");
        let mut response = String::new();
        for token in rx {
            eprint!("{token}");
            response.push_str(&token);
        }
        eprintln!();
        history.push((prompt.to_string(), response));
    }
}
