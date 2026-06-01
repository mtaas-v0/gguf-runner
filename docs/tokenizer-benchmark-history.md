# Tokenizer Benchmark History

Append new results with:

```sh
docs/run-tokenizer-bench.sh
```

Each entry records:

- UTC timestamp
- current git revision
- release-mode synthetic benchmark summaries for GPT-2-style and
  SentencePiece-style tokenization

## 2026-06-01T19:50:18Z (3da9e17)

| Mode | Docs | Bytes/Doc | Ref Min us | Ref Median us | Ref Max us | Opt Min us | Opt Median us | Opt Max us | Median Speedup x |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2 | 8 | 6480 | 10475 | 11192 | 13640 | 5763 | 5813 | 5831 | 1.9251 |
| sentencepiece | 2 | 2160 | 85997 | 86510 | 87500 | 27055 | 27166 | 27543 | 3.1845 |

## 2026-06-01T19:55:40Z (3da9e17)

| Mode | Docs | Bytes/Doc | Ref Min us | Ref Median us | Ref Max us | Opt Min us | Opt Median us | Opt Max us | Median Speedup x |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2 | 8 | 6480 | 10225 | 11371 | 16935 | 5761 | 5847 | 6052 | 1.9445 |
| sentencepiece | 2 | 2160 | 88547 | 89181 | 90172 | 234 | 235 | 293 | 378.2226 |

## 2026-06-01T19:58:33Z (3da9e17)

| Mode | Docs | Bytes/Doc | Ref Min us | Ref Median us | Ref Max us | Opt Min us | Opt Median us | Opt Max us | Median Speedup x |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2 | 8 | 6480 | 10679 | 11285 | 16797 | 5712 | 5748 | 5907 | 1.9631 |
| sentencepiece | 2 | 2160 | 87080 | 87477 | 88408 | 244 | 249 | 303 | 350.9068 |
| gpt2_chunk_1k | 24 | 1024 | 5152 | 5174 | 5203 | 2719 | 2801 | 2995 | 1.8471 |
| gpt2_chunk_2k_code | 16 | 2048 | 6575 | 6651 | 7016 | 3854 | 3896 | 4079 | 1.7071 |
| sentencepiece_chunk_1k | 6 | 1024 | 58418 | 58630 | 59301 | 309 | 323 | 338 | 181.4939 |
| sentencepiece_chunk_2k_code | 4 | 2048 | 157850 | 158361 | 161367 | 479 | 496 | 518 | 319.1165 |

## 2026-06-01T20:03:54Z (3da9e17)

| Mode | Docs | Bytes/Doc | Ref Min us | Ref Median us | Ref Max us | Opt Min us | Opt Median us | Opt Max us | Median Speedup x |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2 | 8 | 6480 | 10653 | 10943 | 13454 | 5699 | 5743 | 5795 | 1.9052 |
| sentencepiece | 2 | 2160 | 86449 | 87115 | 95188 | 268 | 275 | 319 | 316.1126 |
| gpt2_chunk_1k | 24 | 1024 | 5240 | 5250 | 13759 | 2751 | 3108 | 4195 | 1.6895 |
| gpt2_chunk_2k_code | 16 | 2048 | 6787 | 7226 | 10787 | 3793 | 3893 | 4321 | 1.8561 |
| sentencepiece_chunk_1k | 6 | 1024 | 58790 | 59190 | 64886 | 348 | 348 | 354 | 169.8430 |
| sentencepiece_chunk_2k_code | 4 | 2048 | 159159 | 160280 | 165745 | 470 | 476 | 502 | 336.0777 |
| gpt2_source_chunk_1200 | 387 | 1200 | 88460 | 89257 | 89672 | 45016 | 45136 | 45503 | 1.9775 |
| gpt2_source_chunk_1800 | 259 | 1800 | 88448 | 89134 | 100289 | 44960 | 45089 | 45574 | 1.9768 |
| sentencepiece_source_chunk_1200 | 387 | 1200 | 3829154 | 3895432 | 4553271 | 34115 | 34288 | 34724 | 113.6088 |
| sentencepiece_source_chunk_1800 | 259 | 1800 | 5775353 | 5796434 | 6219963 | 34711 | 34934 | 36494 | 165.9217 |

## 2026-06-01T20:34:53Z (3da9e17)

| Mode | Docs | Bytes/Doc | Ref Min us | Ref Median us | Ref Max us | Opt Min us | Opt Median us | Opt Max us | Median Speedup x |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2 | 8 | 6480 | 10143 | 11290 | 16638 | 5829 | 5890 | 6025 | 1.9168 |
| sentencepiece | 2 | 2160 | 86161 | 86580 | 89060 | 254 | 260 | 306 | 332.3631 |
| gpt2_chunk_1k | 24 | 1024 | 5258 | 5298 | 5509 | 2715 | 2785 | 2883 | 1.9021 |
| gpt2_chunk_2k_code | 16 | 2048 | 6536 | 6601 | 6764 | 3866 | 3914 | 4065 | 1.6867 |
| sentencepiece_chunk_1k | 6 | 1024 | 58656 | 58894 | 62342 | 314 | 316 | 327 | 185.8601 |
| sentencepiece_chunk_2k_code | 4 | 2048 | 159282 | 159804 | 160735 | 511 | 520 | 693 | 307.2913 |
| gpt2_source_chunk_1200 | 387 | 1200 | 88226 | 88497 | 89742 | 45122 | 45467 | 46005 | 1.9464 |
| gpt2_source_chunk_1800 | 259 | 1800 | 87645 | 88266 | 90313 | 44838 | 45097 | 45907 | 1.9572 |
| sentencepiece_source_chunk_1200 | 387 | 1200 | 3788753 | 3795575 | 3812219 | 33724 | 34050 | 34495 | 111.4683 |
| sentencepiece_source_chunk_1800 | 259 | 1800 | 5736895 | 5743763 | 5764916 | 35255 | 35323 | 35380 | 162.6057 |
