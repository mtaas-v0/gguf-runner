# RAG Chunk Dedup History

Append new results with:

```sh
docs/run-rag-dedup-analysis.sh
```

Each entry records:

- UTC timestamp
- current git revision
- release-mode dedup summaries for source-derived chunk sets

## 2026-06-01T20:43:05Z (3da9e17)

| Source Dir | Source Files | Max Chars | Chunks | Unique Chunks | Duplicate Chunks | Duplicate Ratio | Reused Texts | Max Reuse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| /Users/jens/tmp/everlock/docs | 45 | 1200 | 1661 | 1467 | 194 | 0.1168 | 181 | 7 |
| /Users/jens/tmp/everlock/docs | 45 | 1800 | 851 | 755 | 96 | 0.1128 | 87 | 4 |
