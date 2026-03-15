pub(crate) mod chunker;
pub(crate) mod encoder;
pub(crate) mod index_io;

use crate::engine::kernels::matmul_f32_embeddings;
use crate::engine::runtime::malloc_run_state;
use crate::engine::types::RunState;
use encoder::{BertPrefillState, embed_raw};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

pub(crate) use encoder::DocumentEncoder;
type EmbedBatchWithStatsResult = Result<(Vec<(usize, Vec<f32>)>, RagBuildWorkerStats), String>;

#[derive(Debug, Clone)]
struct RagBuildWorkerStats {
    worker_idx: usize,
    chunks: usize,
    tokens: usize,
    max_tokens: usize,
    est_cost: usize,
    elapsed_secs: f64,
}

#[derive(Debug, Clone)]
struct RagBuildTrace {
    chunks: usize,
    dim: usize,
    is_bert_family: bool,
    total_tokens: usize,
    avg_tokens_per_chunk: f64,
    max_tokens: usize,
    chunking_secs: f64,
    tokenization_secs: f64,
    scheduling_secs: f64,
    state_alloc_secs: f64,
    embedding_secs: f64,
    assembly_secs: f64,
    total_secs: f64,
    worker_stats: Vec<RagBuildWorkerStats>,
}

impl RagBuildTrace {
    fn print(&self) {
        let arch = if self.is_bert_family {
            "bert-family"
        } else {
            "decoder-family"
        };
        eprintln!(
            "[RAG-BUILD] chunks={} dim={} arch={} tokens={} avg_tokens/chunk={:.1} max_tokens={}",
            self.chunks,
            self.dim,
            arch,
            self.total_tokens,
            self.avg_tokens_per_chunk,
            self.max_tokens
        );
        eprintln!(
            "[RAG-BUILD] chunking={:.3}s tokenization={:.3}s scheduling={:.3}s state_alloc={:.3}s embedding={:.3}s assembly={:.3}s total={:.3}s",
            self.chunking_secs,
            self.tokenization_secs,
            self.scheduling_secs,
            self.state_alloc_secs,
            self.embedding_secs,
            self.assembly_secs,
            self.total_secs
        );
        for worker in &self.worker_stats {
            let tok_per_sec = if worker.elapsed_secs > 0.0 {
                worker.tokens as f64 / worker.elapsed_secs
            } else {
                0.0
            };
            eprintln!(
                "[RAG-BUILD] worker={} chunks={} tokens={} max_tokens={} est_cost={} elapsed={:.3}s tok/s={:.0}",
                worker.worker_idx,
                worker.chunks,
                worker.tokens,
                worker.max_tokens,
                worker.est_cost,
                worker.elapsed_secs,
                tok_per_sec
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(crate) struct RagChunk {
    /// Relative path of the source file within the wiki, e.g. `ops/deploy.md`.
    pub(crate) source: String,
    /// The chunk text injected verbatim into the system prompt.
    pub(crate) text: String,
    /// L2-normalised embedding vector (length = index `dim`).
    pub(crate) embedding: Vec<f32>,
}

// ---------------------------------------------------------------------------
// RagIndex
// ---------------------------------------------------------------------------

/// In-memory RAG index: a flat list of chunks with a row-major embedding matrix
/// for fast batched cosine similarity search.
pub(crate) struct RagIndex {
    chunks: Vec<RagChunk>,
    dim: usize,
    /// Row-major [n_chunks × dim] matrix — ready for `matmul_f32_embeddings`.
    flat_embeddings: Vec<f32>,
}

impl RagIndex {
    /// Build an index from `wiki_dir`, embedding each chunk with `encoder`.
    /// Prints progress to stderr.
    pub(crate) fn build_from_dir(
        wiki_dir: &std::path::Path,
        encoder: &mut DocumentEncoder,
        max_chars_per_chunk: usize,
        max_tokens_per_chunk: usize,
        progress: Option<std::sync::Arc<dyn Fn(String) + Send + Sync>>,
        trace_build: bool,
    ) -> Result<Self, String> {
        let build_started = Instant::now();
        let chunk_t0 = Instant::now();
        let raw_chunks = chunker::chunk_directory(wiki_dir, max_chars_per_chunk)?;
        let chunking_secs = chunk_t0.elapsed().as_secs_f64();
        let total = raw_chunks.len();
        if total == 0 {
            return Err(format!(
                "no markdown files found under '{}'",
                wiki_dir.display()
            ));
        }
        let dim = encoder.dim();

        // ── Phase 1: tokenise all chunks sequentially ──────────────────────
        // `bpe_encode` requires `&mut Tokenizer` (lazy hashmap init), so this
        // cannot be parallelised without cloning the tokenizer per thread.
        // Tokenisation is fast compared to inference; do it upfront.
        let token_t0 = Instant::now();
        let all_token_ids: Vec<Vec<i32>> = {
            let doc_prefix = encoder.document_prefix().to_string();
            let mut out = Vec::with_capacity(total);
            let mut ids = Vec::new();
            for raw in &raw_chunks {
                let text = if doc_prefix.is_empty() {
                    raw.text.clone()
                } else {
                    format!("{doc_prefix}{}", raw.text)
                };
                encoder.tokenize(&text, &mut ids);
                if max_tokens_per_chunk > 0 && ids.len() > max_tokens_per_chunk {
                    ids.truncate(max_tokens_per_chunk);
                }
                out.push(ids.clone());
                ids.clear();
            }
            out
        };
        let tokenization_secs = token_t0.elapsed().as_secs_f64();
        let total_tokens: usize = all_token_ids.iter().map(Vec::len).sum();
        let max_tokens = all_token_ids.iter().map(Vec::len).max().unwrap_or(0);
        let avg_tokens_per_chunk = total_tokens as f64 / total as f64;

        let ctx = encoder.embed_context();
        let done = AtomicUsize::new(0);
        let start = Instant::now();
        let num_threads = rayon::current_num_threads().min(total).max(1);
        let scheduling_secs = 0.0;
        let chunk_size = total.div_ceil(num_threads);

        // ── Phase 2: embed chunks in parallel ──────────────────────────────
        // Each rayon worker gets its own RunState and BertPrefillState so there
        // is no sharing or locking during inference.  The BertPrefillState buffers
        // grow to the largest chunk seen by that worker and are reused for every
        // subsequent chunk, eliminating per-call heap allocation.
        //
        // The serial parts of embed_prefill_bert (attention, RoPE, LayerNorm)
        // are not rayon-parallelised internally, so keeping the outer parallel
        // loop is essential — it is what puts those serial sections on different
        // cores concurrently.
        let state_alloc_t0 = Instant::now();
        let mut workers: Vec<(RunState, BertPrefillState)> = (0..num_threads)
            .map(|_| malloc_run_state(ctx.config).map(|rs| (rs, BertPrefillState::new())))
            .collect::<Result<Vec<_>, _>>()?;
        let state_alloc_secs = state_alloc_t0.elapsed().as_secs_f64();

        let embed_t0 = Instant::now();
        let batches: Vec<EmbedBatchWithStatsResult> = workers
            .par_iter_mut()
            .enumerate()
            .map(|(worker_idx, (rs, bs))| {
                let worker_started = Instant::now();
                let lo = worker_idx * chunk_size;
                let hi = (lo + chunk_size).min(total);
                let batch = all_token_ids.get(lo..hi).unwrap_or(&[]);
                let batch_chunks = batch.len();
                let batch_tokens: usize = batch.iter().map(Vec::len).sum();
                let batch_max_tokens = batch.iter().map(Vec::len).max().unwrap_or(0);
                batch
                    .into_iter()
                    .enumerate()
                    .map(|(offset, ids)| {
                        let idx = lo + offset;
                        let emb = embed_raw(ids, &ctx, rs, bs)?;
                        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                        if n.is_multiple_of(10) || n == total {
                            let elapsed = start.elapsed().as_secs_f64();
                            let rate = n as f64 / elapsed;
                            let eta = ((total - n) as f64 / rate) as u64;
                            let msg = format!("{n}/{total}  {rate:.0} chunks/s  ETA {eta}s");
                            if let Some(cb) = &progress {
                                cb(msg);
                            } else {
                                eprint!("\r\x1b[2K  {msg}");
                            }
                        }
                        Ok((idx, emb))
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map(|results| {
                        (
                            results,
                            RagBuildWorkerStats {
                                worker_idx,
                                chunks: batch_chunks,
                                tokens: batch_tokens,
                                max_tokens: batch_max_tokens,
                                est_cost: 0,
                                elapsed_secs: worker_started.elapsed().as_secs_f64(),
                            },
                        )
                    })
            })
            .collect();
        let embedding_secs = embed_t0.elapsed().as_secs_f64();

        if progress.is_none() {
            eprint!("\r\x1b[2K");
        }

        let assembly_t0 = Instant::now();
        let mut embeddings: Vec<Option<Vec<f32>>> = (0..total).map(|_| None).collect();
        let mut worker_stats = Vec::with_capacity(num_threads);
        for (batch, stats) in batches.into_iter().collect::<Result<Vec<_>, _>>()? {
            for (idx, embedding) in batch {
                embeddings[idx] = Some(embedding);
            }
            worker_stats.push(stats);
        }

        let mut chunks: Vec<RagChunk> = Vec::with_capacity(total);
        let mut flat_embeddings: Vec<f32> = Vec::with_capacity(total * dim);
        for (raw, embedding) in raw_chunks.into_iter().zip(embeddings.into_iter()) {
            let embedding = embedding.ok_or_else(|| "missing RAG embedding result".to_string())?;
            flat_embeddings.extend_from_slice(&embedding);
            chunks.push(RagChunk {
                source: raw.source,
                text: raw.text,
                embedding,
            });
        }
        let assembly_secs = assembly_t0.elapsed().as_secs_f64();

        if trace_build {
            worker_stats.sort_unstable_by(|a, b| {
                b.elapsed_secs
                    .partial_cmp(&a.elapsed_secs)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.worker_idx.cmp(&b.worker_idx))
            });
            RagBuildTrace {
                chunks: total,
                dim,
                is_bert_family: ctx.config.is_bert_family,
                total_tokens,
                avg_tokens_per_chunk,
                max_tokens,
                chunking_secs,
                tokenization_secs,
                scheduling_secs,
                state_alloc_secs,
                embedding_secs,
                assembly_secs,
                total_secs: build_started.elapsed().as_secs_f64(),
                worker_stats,
            }
            .print();
        }

        Ok(Self {
            chunks,
            dim,
            flat_embeddings,
        })
    }

    /// Load a pre-built index from a `.ragidx` file.
    pub(crate) fn load(path: &std::path::Path) -> Result<Self, String> {
        let (dim, chunks) = index_io::load(path)?;
        let flat_embeddings = chunks
            .iter()
            .flat_map(|c| c.embedding.iter().copied())
            .collect();
        Ok(Self {
            chunks,
            dim,
            flat_embeddings,
        })
    }

    /// Persist the index to a `.ragidx` file.
    pub(crate) fn save(&self, path: &std::path::Path) -> Result<(), String> {
        index_io::save(path, self.dim, &self.chunks)
    }

    /// Number of chunks in the index.
    pub(crate) fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Retrieve the `top_k` most similar chunks for `query_embedding`, augmented with
    /// keyword rescue for any significant query words not captured by semantic search.
    ///
    /// `query_embedding` must be L2-normalised (as produced by `DocumentEncoder::embed`).
    /// `query_text` is the raw query string used for keyword matching.
    /// Returns chunks sorted by descending cosine similarity (semantic results first,
    /// then any keyword-rescued chunks that didn't make the semantic top-k).
    pub(crate) fn query(
        &self,
        query_embedding: &[f32],
        query_text: &str,
        top_k: usize,
    ) -> Vec<&RagChunk> {
        if self.chunks.is_empty() || top_k == 0 {
            return Vec::new();
        }
        let n = self.chunks.len();
        let k = top_k.min(n);
        let mut scores = vec![0f32; n];
        // All embeddings are L2-normalised → dot product == cosine similarity.
        matmul_f32_embeddings(
            &mut scores,
            query_embedding,
            &self.flat_embeddings,
            n,
            self.dim,
        );

        // Semantic top-k.
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_unstable_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut selected: Vec<usize> = indices[..k].to_vec();
        let selected_set: std::collections::HashSet<usize> = selected.iter().copied().collect();

        // Keyword rescue: find chunks containing significant query words that semantic
        // search missed.  Limits to at most `k` additional chunks.
        let keywords = extract_keywords(query_text);
        if !keywords.is_empty() {
            let mut kw_scored: Vec<(usize, usize)> = self
                .chunks
                .iter()
                .enumerate()
                .filter(|(i, _)| !selected_set.contains(i))
                .filter_map(|(i, chunk)| {
                    let lower = chunk.text.to_ascii_lowercase();
                    let hits = keywords
                        .iter()
                        .filter(|kw| lower.contains(kw.as_str()))
                        .count();
                    if hits > 0 { Some((i, hits)) } else { None }
                })
                .collect();
            kw_scored.sort_unstable_by(|a, b| b.1.cmp(&a.1));
            for (i, _) in kw_scored.into_iter().take(k) {
                selected.push(i);
            }
        }

        selected.iter().map(|&i| &self.chunks[i]).collect()
    }
}

/// Extract significant keywords from a query string.
/// Strips punctuation, lowercases, and removes common English stop words.
fn extract_keywords(query: &str) -> Vec<String> {
    const STOP_WORDS: &[&str] = &[
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can",
        "need", "dare", "ought", "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
        "they", "them", "their", "this", "that", "these", "those", "what", "who", "which", "where",
        "when", "why", "how", "and", "or", "but", "not", "no", "nor", "so", "yet", "for", "in",
        "on", "at", "to", "of", "by", "as", "with", "from", "about", "into", "than", "then",
        "there",
    ];
    query
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .map(|w| w.to_ascii_lowercase())
        .filter(|w| w.len() >= 3 && !STOP_WORDS.contains(&w.as_str()))
        .collect()
}

// ---------------------------------------------------------------------------
// Context injection helpers
// ---------------------------------------------------------------------------

/// Prepend retrieved chunks to `system_prompt` as a `<knowledge>` block.
pub(crate) fn prepend_rag_context(chunks: &[&RagChunk], system_prompt: &str) -> String {
    if chunks.is_empty() {
        return system_prompt.to_string();
    }

    let mut block = String::from("<knowledge>\n");
    for chunk in chunks {
        block.push('[');
        block.push_str(&chunk.source);
        block.push_str("]\n");
        block.push_str(&chunk.text);
        block.push_str("\n\n");
    }
    // Trim the trailing blank line inside the tag.
    let block = block.trim_end_matches('\n');
    let block = format!("{block}\n</knowledge>");

    if system_prompt.is_empty() {
        block
    } else {
        format!("{block}\n\n{system_prompt}")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(source: &str, text: &str, emb: Vec<f32>) -> RagChunk {
        RagChunk {
            source: source.to_string(),
            text: text.to_string(),
            embedding: emb,
        }
    }

    fn unit(v: Vec<f32>) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.into_iter().map(|x| x / norm).collect()
    }

    #[test]
    fn query_returns_top_k() {
        let dim = 4;
        let chunks = vec![
            make_chunk("a.md", "alpha", unit(vec![1.0, 0.0, 0.0, 0.0])),
            make_chunk("b.md", "beta", unit(vec![0.0, 1.0, 0.0, 0.0])),
            make_chunk("c.md", "gamma", unit(vec![0.0, 0.0, 1.0, 0.0])),
        ];
        let flat: Vec<f32> = chunks.iter().flat_map(|c| c.embedding.clone()).collect();
        let index = RagIndex {
            chunks,
            dim,
            flat_embeddings: flat,
        };
        let query = unit(vec![1.0, 0.1, 0.0, 0.0]);
        let results = index.query(&query, "alpha test", 2);
        assert_eq!(results.len(), 2);
        // "alpha" should rank first.
        assert_eq!(results[0].source, "a.md");
    }

    #[test]
    fn prepend_context_format() {
        let chunk = RagChunk {
            source: "ops/deploy.md".to_string(),
            text: "Deploy with ./deploy.sh".to_string(),
            embedding: vec![],
        };
        let out = prepend_rag_context(&[&chunk], "Be helpful.");
        assert!(out.starts_with("<knowledge>"));
        assert!(out.contains("[ops/deploy.md]"));
        assert!(out.contains("Deploy with ./deploy.sh"));
        assert!(out.contains("</knowledge>"));
        assert!(out.contains("Be helpful."));
    }

    #[test]
    fn prepend_context_no_chunks() {
        let out = prepend_rag_context(&[], "original");
        assert_eq!(out, "original");
    }
}
