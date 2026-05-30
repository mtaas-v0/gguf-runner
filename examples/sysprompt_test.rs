use std::fs;

fn main() {
    let path = "./Qwen3.5-2B-Q4_K_M.gguf";
    let data = fs::read(path).expect("model file");
    let data: &'static [u8] = Box::leak(data.into_boxed_slice());

    let prompts = [
        "You are a helpful assistant.",
        "You are Everlock AI.",
        "You are Everlock AI. Answer clearly.",
        "You are Everlock AI. Answer clearly and concisely for an operator.",
        "You are Everlock AI. Answer clearly and concisely for an operator using the Everlock admin shell.",
    ];

    let question = "What is the capital of France?";

    for (i, system) in prompts.iter().enumerate() {
        eprintln!("\n=== Variant {} ===\nsystem: \"{system}\"", i + 1);
        let mut rt = gguf_runner::EmbeddedRuntime::load_from_bytes(data).expect("load");
        let rx = rt.generate(&[], question, system).expect("generate");
        let response: String = rx.into_iter().collect();
        eprintln!("response: \"{response}\"");
    }
}
