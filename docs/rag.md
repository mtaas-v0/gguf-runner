# RAG (Retrieval-Augmented Generation)

RAG lets the model answer questions about your own documents — wikis, runbooks, internal notes — without fine-tuning. A second embedding model (sidecar) converts text into vectors; at query time the most relevant chunks are retrieved and injected into the system prompt before the model sees your question.

---

## What you need

### 1. A language model

Any GGUF you already use. The RAG pipeline is additive — it only prepends a `<knowledge>` block to the system prompt; the model itself is untouched.

### 2. An embedding sidecar GGUF

A separate, small embedding model that converts text to vectors. Recommended:

- **nomic-embed-text-v1.5** (Q4_K_M, ~85 MB) — strong retrieval, instruction-tuned for `search_query:` / `search_document:` prefixes, auto-detected by architecture
- **all-MiniLM-L6-v2** — fast, lightweight, good for general text

Download example:
```sh
wget https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf
```

The encoder does **not** need to match the language model family. It only needs to be a GGUF with a transformer architecture (`bert`, `nomic-bert`, `roberta`, or any decoder-only model that supports mean pooling).

### 3. A document corpus

A directory tree of Markdown files (`.md`). Subdirectories are walked recursively. The chunker splits on `#`/`##`/`###` headers and further splits large sections with ~20% overlap at paragraph boundaries.

---

## Index file format (`.ragidx`)

The index is a compact binary file:

```
[8  bytes]  magic: RAGIDX v1
[4  bytes]  embedding_dim (u32 LE)
[4  bytes]  chunk_count (u32 LE)
per chunk:
  [2 bytes]  source path length (u16 LE)
  [N bytes]  source path (UTF-8, relative to wiki root)
  [4 bytes]  text length (u32 LE)
  [N bytes]  chunk text (UTF-8)
  [dim×4]    embedding (f32 LE array)
```

**Estimated sizes** (nomic-embed, dim=768):

| Chunks | Index size |
|--------|-----------|
| 1,000  | ~4 MB     |
| 10,000 | ~38 MB    |
| 40,000 | ~150 MB   |

Loading a 150 MB index from SSD takes under a second.

---

## CLI usage

### Step 1 — Build the index (one-time)

```sh
gguf-runner \
  --model       Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --rag-encoder nomic-embed-text-v1.5.Q4_K_M.gguf \
  --rag-source  /path/to/wiki \
  --rag-index   /path/to/wiki.ragidx \
  --rag-max-chars-per-chunk 1200 \
  --rag-max-tokens-per-chunk 320 \
  --rag-build
```

- `--rag-build` builds the index, saves it, and exits — no inference, no `--prompt` required.
- Progress is printed to stderr: `n/total  rate chunks/s  ETA Xs`.
- Rebuild anytime by re-running the same command (overwrites the file).

### Step 2 — Query (oneshot)

```sh
gguf-runner \
  --model       Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --rag-encoder nomic-embed-text-v1.5.Q4_K_M.gguf \
  --rag-index   /path/to/wiki.ragidx \
  --prompt      "How do I roll back a deployment?"
```

`--rag-source` is not needed if the index file already exists.

### Step 3 — Interactive (REPL)

```sh
gguf-runner \
  --model       Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --rag-encoder nomic-embed-text-v1.5.Q4_K_M.gguf \
  --rag-index   /path/to/wiki.ragidx \
  --mode        repl
```

Every message you type will automatically retrieve and inject the top-k most relevant chunks before the model responds.

### With agent tools

```sh
gguf-runner \
  --model       Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --rag-encoder nomic-embed-text-v1.5.Q4_K_M.gguf \
  --rag-index   /path/to/wiki.ragidx \
  --mode        repl \
  --tools
```

When `--tools` is enabled together with a loaded index, the model also gets a `search_knowledge` tool it can call proactively during multi-turn conversations to fetch additional context it needs.

### All RAG flags

| Flag | Default | Description |
|------|---------|-------------|
| `--rag-encoder <path>` | auto-discover | Path to embedding sidecar GGUF |
| `--rag-index <path>` | — | Path to `.ragidx` file (load or save) |
| `--rag-source <dir>` | — | Wiki source directory (build from here) |
| `--rag-top-k <n>` | `5` | Chunks to inject per turn |
| `--rag-max-chars-per-chunk <n>` | `1800` | Soft character limit per indexed chunk |
| `--rag-max-tokens-per-chunk <n>` | `0` | Optional token cap per indexed chunk after tokenization (`0` disables the cap) |
| `--rag-build` | false | Build index and exit (no inference) |

---

## REPL commands

Once in REPL mode, the RAG index can be managed live without restarting:

| Command | Description |
|---------|-------------|
| `/doc <path>` | Build and load a RAG index from a wiki directory. Uses the encoder specified at startup. Progress appears in the status bar. |
| `/docs` | Show current RAG status: number of chunks loaded, encoder path. |
| `/clear-docs` | Unload the active RAG index. |

Example session:
```
> /doc /path/to/wiki
[sys] Building knowledge index…          ← status bar while indexing
[sys] RAG: 3842 chunks loaded from '/path/to/wiki'

> what was the apimeister budget in 2025?
[sys] Retrieved 5 chunks from knowledge base
…model response using injected context…

> /docs
[sys] RAG index active: 3842 chunks loaded

> /clear-docs
[sys] RAG index cleared.
```

---

## Encoder auto-discovery

If `--rag-encoder` is not specified, the runner probes the model's directory for sidecars in this order:

1. `embed-{model-filename}`  — e.g. `embed-Qwen3-35B-Q4.gguf`
2. `{model-stem}.embed.gguf`
3. `encoder.gguf`
4. Any `embed*.gguf` or `encoder*.gguf` in the same directory (alphabetically first)

Place your embedding GGUF in the same directory as the model and it will be picked up automatically.

---

## How retrieval works

Each turn:

1. The last user message is embedded using the encoder sidecar.
2. Cosine similarity is computed against all indexed chunk vectors (dot product of L2-normalised vectors via the same `matmul_f32_embeddings` kernel used for transformer inference).
3. The top-k chunks are selected by score.
4. **Keyword rescue**: significant words (≥3 chars, not stop words) from the query are matched against chunk text; any chunks containing those words that weren't already in the top-k are appended (up to k additional chunks). This catches proper nouns and exact identifiers that semantic search can miss.
5. Retrieved chunks are prepended to the system prompt as a `<knowledge>` block:

```
<knowledge>
[ops/deploy.md]
Deployments are triggered by pushing to main…

[ops/rollback.md]
To roll back, run: ./scripts/rollback.sh <version>…
</knowledge>

{your system prompt}
```

With `--tools`, the model can additionally call `search_knowledge` mid-conversation to fetch more context:
```json
{"type":"tool_call","tool":"search_knowledge","args":{"query":"budget 2025","top_k":5}}
```

---

## nomic-embed-text task prefixes

`nomic-embed-text-v1.5` uses instruction-tuned embeddings and requires prefixes for retrieval to work correctly:

- Queries are prefixed with `search_query: `
- Documents are prefixed with `search_document: ` at index-build time

This is handled automatically when the encoder's `general.architecture` is `nomic-bert`. No user action required.

---

## Performance notes

- **Index build**: exact duplicate chunk texts are deduplicated first, then unique chunks go through parallel tokenisation and parallel transformer inference across rayon workers (one `RunState` per thread). For BERT-family encoders, smaller chunks can be materially faster because attention cost grows with sequence length, but smaller chunks also raise total chunk count. A token cap is often a better tradeoff: try `--rag-max-tokens-per-chunk 256` or `320` before reducing `--rag-max-chars-per-chunk` aggressively.
- **Index load**: sequential disk read, ~1 s for 150 MB on SSD.
- **Query time**: negligible — a single matrix-vector multiply over the flat embedding matrix, parallelised by rayon.
- **Memory**: the flat embedding matrix for 40k × 768 = ~117 MB resident in RAM while loaded.
- **Encoder seq_len**: capped at 2048 tokens at load time regardless of the model's native context window, to keep per-thread `RunState` allocations small during index building.
