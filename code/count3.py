"""count3: same as count2 but extractor = EXM (default Opus 4.6) with status field; baseline = plain RRF run of READER.
Two-pass count for 'how many / how much / total' questions (selected by regex on question text; no gold, no question_type).
Pass 1: gpt-5-mini reads EACH top-5 session alone and lists every item/event/amount matching the question (JSON: item, date, quote, status).
Pass 2: the reader (gpt-5) gets the normal official prompt + sessions, plus the per-session item lists as notes, and must
dedupe across sessions, apply the question's filters (time window, category, user's own actions, latest updates), then count."""
import json, os, re, collections
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault("SRR", "rerank_schema_all_nothink.json"); os.environ.setdefault("READER", "claude-opus-4-6"); EXM = os.environ.get("EXM", "claude-opus-4-6")
exec(open("official_A.py").read().split("with ThreadPoolExecutor")[0])
CNT = re.compile(r"\bhow many\b|\bhow much\b|\btotal\b|\bcombined\b|\baltogether\b", re.I)
QIDS = [x for x in QIDS if CNT.search(D[x]["question"])]
P1 = ("Below is ONE past chat session between a user and an assistant, and a question the user is asking now (on {}).\n"
      "List EVERY item, event, purchase, amount or occurrence in this session that could count toward answering the question, "
      "even if unsure. Mention it even if only said in passing. For each give: item, what happened, the date or time if stated (session date {}), "
      "the user's exact words, and status (completed / planned / cancelled / hypothetical / someone else's), and whether it updates/replaces an earlier value.\n"
      "Return a JSON list only (empty list if nothing).\n\nQuestion: {}\n\nSession:\n{}")
NOTE = ("\n\nNotes: a helper read each session separately and listed candidate items below. They may include duplicates (the same thing mentioned in several sessions), "
        "items that do not match the question, or miss things. Use them as a checklist: verify each against the chats, merge duplicates, "
        "apply every condition in the question (time window, category, the user's own actions, latest updates), then count.\n{}")
def one(qid):
    q = D[qid]; ch = sorted(top(qid), key=lambda x: x[0])
    lists = list(ThreadPoolExecutor(5).map(lambda p: call(EXM, P1.format(q["question_date"], p[0], q["question"], json.dumps(p[1])), 4000), ch))
    notes = "".join(f"\nSession {i + 1} candidates: {l.strip()[:3000]}" for i, l in enumerate(lists))
    hs = "".join("\n### Session {}:\nSession Date: {}\nSession Content:\n\n{}\n".format(i + 1, sd, json.dumps(ss)) for i, (sd, ss) in enumerate(ch))
    a = call(READER, TPL.format(hs + NOTE.format(notes), q["question_date"], q["question"]), 800)
    j = call("gpt-4o-2024-08-06", ev.get_anscheck_prompt(q["question_type"], q["question"], q["answer"], a, abstention="_abs" in qid), 10)
    return dict(qid=qid, qt=q["question_type"], ans=a, notes=notes, ok="yes" in j.lower())
with ThreadPoolExecutor(int(os.environ.get("PAR", "3"))) as ex: R = list(ex.map(one, QIDS))
open(f"official_CNT3{EXM.split('-')[1]}_{READER}_k5_{SET}.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in R))
base = {json.loads(l)["qid"]: json.loads(l)["ok"] for l in open(f"official_RRF_{READER}_json_k5_{SET}.jsonl")}
fx = [r for r in R if r["ok"] and not base[r["qid"]]]; br = [r for r in R if base[r["qid"]] and not r["ok"]]
print(f"two-pass count {READER} {SET}: {len(R)} count-style questions | baseline {sum(base[r['qid']] for r in R)} -> {sum(r['ok'] for r in R)} | fixed {len(fx)} broke {len(br)}")
print(f"full {SET}: {sum(base.values())} -> {sum(base.values()) - sum(base[r['qid']] for r in R) + sum(r['ok'] for r in R)}/{len(base)}")
for r in fx: print("  FIXED", D[r["qid"]]["question"][:90])
for r in br: print("  BROKE", D[r["qid"]]["question"][:90], "| GOLD", str(D[r["qid"]]["answer"])[:40])
print("tokens", dict(TOK), flush=True)
