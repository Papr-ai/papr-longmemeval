# Papr on LongMemEval-S — 95.4% (Papr High) / 94.4% (Papr Low)

Per-question answers, official grading logs, retrieved sessions and code for Papr memory retrieval on
[LongMemEval](https://github.com/xiaowu0162/LongMemEval) (`longmemeval_s_cleaned.json`, 500 questions).

## Results (official `evaluate_qa.py`, judge `gpt-4o-2024-08-06`, unmodified prompts)

| Setup | Answer model | Overall | Task-avg | Abstention | Context to answer model |
|---|---|---|---|---|---|
| **Papr High** | Claude Opus 4.6 (8k thinking) | **95.4% (477/500)** | 96.94% | 27/30 | ~16.5k tokens/q |
| **Papr Low** | gpt-5-mini | **94.4% (472/500)** | 95.85% | 25/30 | ~15k tokens/q |

Per type (correct / total, abstention questions counted separately):

| Type | Papr High | Papr Low |
|---|---|---|
| single-session-user | 64/64 | 63/64 |
| single-session-assistant | 56/56 | 56/56 |
| single-session-preference | 30/30 | 30/30 |
| knowledge-update | 70/72 | 67/72 |
| temporal-reasoning | 118/127 | 119/127 |
| multi-session | 112/121 | 112/121 |
| abstention | 27/30 | 25/30 |

Reproduce the scores from the files here:
`python src/evaluation/print_qa_metrics.py results/<name>.jsonl.eval-results-gpt-4o data/longmemeval_s_cleaned.json`

## Pipeline

1. **Embed** every user turn and each question with `papr-embed-schema-1-0.6b` (Papr embeddings API, schema `memory:personal:1.6.0`). A session's score is its best turn's cosine similarity to the question. Top 20 sessions kept. (`code/embed_schema_all.py`)
2. **Rerank** the 20 with Gemini 2.5 Flash (thinking off): one call per question grades each session for whether it contains evidence needed to answer; order = grade, then embedding rank. (`code/rerank_schema.py`, `code/rerank40.py`)
3. **Answer** with the top 5 sessions, sorted by date, using the **official LongMemEval CoT prompt and JSON history format** from `run_generation.py`. (`code/official_A.py`)
   - Papr High only: for count/total-style questions (picked by a regex on the question text, no labels used), Opus first lists matching items per session, then answers with those notes. (`code/count3.py`, `code/route_cnt.py`)
4. **Grade** with the official `evaluate_qa.py` and `gpt-4o-2024-08-06`. (`code/official_eval_submit.py`)

Retrieval quality (all evidence sessions in the top 5, 470 answerable questions): **460/470 (97.9%)**; in the top 20: 467/470.
Reranker latency: 1.8 s median / 2.1 s p90 per question.

## Files

- `results/papr_*.jsonl` — `{question_id, hypothesis}` in the official submission format.
- `results/papr_*.jsonl.eval-results-gpt-4o` — official judge output per question.
- `retrieval/top5_sessions.jsonl` — per question: top-5 and top-20 session ids after reranking, rerank grades, latency.
- `code/` — the scripts above (keys read from environment variables).

## Caveats (please read)

- **Run-to-run variance.** gpt-5-mini: two full runs scored 469 and 472; we report the second. Opus 4.6 + extraction: two runs scored 477 and 473 (mean 475, 95.0%); we report 477. Expect ±3 questions.
- **Dev/test.** We iterated on prompts and routing using half of the 500 questions (248, `dev`) and checked on the other 252. There is no held-out set beyond that, so treat the numbers as tuned on this benchmark like most reported LongMemEval results.
- No question labels (`question_type`, `_abs`, answer sessions) are used anywhere in retrieval or answering; they are only used for grading and reporting.
