## Papr Embed — 96.0% (480/500) on LongMemEval-S, official evaluator, per-question files included

**Repo with all artifacts:** https://github.com/Papr-ai/papr-longmemeval

| Setup | Answer model | Overall | Task-avg | Abstention | Retrieval speed (median, 1 request) | Reader tokens/q |
|---|---|---|---|---|---|---|
| **Papr Embed High, K=20** | Claude Opus 4.6 (8k thinking) | **96.0% (480/500)** | 96.71% | 27/30 | ~1.9 s | not measured (20 sessions vs 5) |
| **Papr Embed High** | Claude Opus 4.6 (8k thinking) | **95.4% (477/500)** | 96.94% | 27/30 | ~1.9 s | ~16.5k |
| **Papr Embed Rerank** | gpt-5-mini | **94.4% (472/500)** | 95.85% | 25/30 | ~1.9 s | ~15k |
| **Papr Embed Low** | gpt-5-mini | **91.0% (455/500)** | 92.4% | 26/30 | ~0.17 s | ~14.2k |

Papr Embed Low = schema embeddings only, no reranker. Rerank and High add the Gemini 2.5 Flash reranker (1.8 s median). Speed = embedding (~170 ms) + search (<1 ms) [+ reranker]; reranker tokens not counted in reader tokens.

- Data: `longmemeval_s_cleaned.json`, all 500 questions, real retrieval over each haystack (no oracle).
- Answering: official CoT prompt and JSON history format from `run_generation.py`, top 5 retrieved sessions.
- Grading: unmodified `evaluate_qa.py` with `gpt-4o-2024-08-06`; the `.eval-results-gpt-4o` files are in the repo, so `print_qa_metrics.py` reproduces the numbers directly.
- Retrieval: Papr schema embeddings (top 20 sessions) + Gemini 2.5 Flash rerank → top 5. All evidence sessions in the top 5 for 460/470 answerable questions.
- Context to the answer model: ~15–16.5k tokens per question.

**Papr Embed High, K=20** gives the reader the top 20 reranked sessions instead of 5 (no extraction step): 480/500, 96.0%; per-question files `results/papr_high_k20_opus-4-6.jsonl*`. Net +3 vs K=5 (11 gained, 8 lost), within the variance below, and K=20 was picked after reviewing K=5's misses.

Caveats: run-to-run variance is about ±3 questions (second runs: 473 for Embed High, 469 for Embed Rerank). Prompts/routing were iterated on half of the questions and checked on the other half. No question labels are used in retrieval or answering.

Happy to answer methodology questions.
