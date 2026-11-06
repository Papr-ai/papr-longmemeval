"""LLM session reranker on top of prev-best (SETUP). Per question: top-K sessions -> digest (date + extracted records + best-matching turns)
-> Gemini grades each session 0-3 for 'contains evidence needed to answer'. Final rank = grade desc, then prior rank. Generic prompt, no tuned examples."""
import json, os, sys, math, pickle, collections, numpy as np
from concurrent.futures import ThreadPoolExecutor
import conv_v3 as C
K = int(os.environ.get("RK", "20")); MODEL = os.environ.get("RM", "gemini-2.5-flash"); TAG = os.environ.get("TAG", "v1")
D = {q["question_id"]: q for q in json.load(open("longmemeval_s_cleaned.json"))}; keys = set(json.load(open("turn_keys.json")))
rows = pickle.load(open(os.environ.get("ROWS", "rows_h512.pkl"), "rb"))
if os.environ.get("TESTX"): rows = [r for r in rows if r["split"] == "test"]
if os.environ.get("DEVX"): _s = set(json.load(open("small_qids.json"))); rows = [r for r in rows if r["split"] == "dev" and r["qid"] not in _s]
if os.environ.get("QIDSF"): _q = set(json.load(open(os.environ["QIDSF"]))); rows = [r for r in rows if r["qid"] in _q]
CANDS = {**json.load(open("rerank_v1.json")), **json.load(open("rerank_devx.json"))} if os.environ.get("CANDS") == "1" else None
import time as _t; LAT = []
W = dict(S=.15, E=.05, P=.03, T=.10, U=.10)
def base(r): return r["dense"] + sum(w * r[b] for b, w in W.items())
def sids_of(q): return [sid for sid, s in zip(q["haystack_session_ids"], q["haystack_sessions"]) if any(f"{sid}|{i}" in keys for i in range(len(s)))]
def srank(r, sc):
    agg = np.full(r["ns"], -1e9); np.maximum.at(agg, r["sidx"], sc); return list(np.argsort(-agg)), agg
def digest(sid, sdate, sess, tscores):
    try: recs = json.load(open(f"facts_v3f/{sid}.json"))["records"]
    except Exception: recs = []
    lines = [f"date: {sdate}"]
    for r in recs[:25]:
        lines.append("- " + " | ".join(str(x) for x in [r.get("event_class"), r.get("subject"), r.get("attribute"), r.get("value"), r.get("event_date")] if x not in (None, "", "other")))
    ut = [t["content"] for t in sess if t["role"] == "user"]
    best = sorted(range(len(sess)), key=lambda i: -tscores.get(i, -9))[:2]
    for i in best: lines.append(f"turn: {sess[i]['content'][:500]}")
    if ut: lines.append(f"user said: {ut[0][:300]}")
    return "\n".join(lines)
SCH = {"type": "OBJECT", "properties": {"grades": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {"id": {"type": "INTEGER"}, "grade": {"type": "INTEGER"}}, "required": ["id", "grade"]}}}, "required": ["grades"]}
FINE = os.environ.get("FINE") == "1"
P = """You rank a user's past chat sessions for a memory system. Question (asked {qd}): {q}
For EACH session below, GRADEHDR
3 = directly contains a fact needed for the answer (any one of several facts needed for counting, summing, comparing, ordering or date arithmetic counts as 3);
2 = likely relevant (same specific topic/entity, possibly an older/newer value of the same thing);
1 = loosely related topic; 0 = unrelated.
Questions that count or total things usually need SEVERAL sessions — grade every session that mentions an instance as 3.
Sessions:
{s}"""
if FINE: P = P.replace("GRADEHDR", "grade 0-10 how much it contains evidence needed to answer the question (use the full range; ties are bad — put the most directly answering sessions highest, e.g. the session stating the exact fact asked, or the most recent value for questions about current state). Anchors:").replace("as 3", "as 8-10")
else: P = P.replace("GRADEHDR", "grade 0-3 how much it contains evidence needed to answer the question:")
cache_f = f"rerank_{TAG}.json"; cache = json.load(open(cache_f)) if os.path.exists(cache_f) else {}
def go(r):
    if r["qid"] in cache: return
    q = D[r["qid"]]; sids = sids_of(q); assert len(sids) == r["ns"], (len(sids), r["ns"])
    sc = base(r); rk, agg = srank(r, sc); top = rk[:K]
    if CANDS: top = CANDS[r["qid"]]["top"][:K]
    bys = collections.defaultdict(dict); ti = collections.Counter()
    # map turn rows to (session, turn index) in same order as build()
    sess_turns = []
    for sid, s in zip(q["haystack_session_ids"], q["haystack_sessions"]):
        for i in range(len(s)):
            if f"{sid}|{i}" in keys: sess_turns.append((sid, i))
    for j, (sid, i) in enumerate(sess_turns): bys[sid][i] = float(sc[j])
    meta = {sid: (s, dd) for sid, s, dd in zip(q["haystack_session_ids"], q["haystack_sessions"], q["haystack_dates"])}
    body = "\n\n".join(f"[{n}]\n" + digest(sids[si], meta[sids[si]][1], meta[sids[si]][0], bys[sids[si]]) for n, si in enumerate(top))
    _t0 = _t.time(); out = C.gem(P.format(qd=q["question_date"], q=q["question"], s=body), SCH, model=MODEL)
    LAT.append(_t.time() - _t0); g = {it["id"]: it["grade"] for it in (out or {}).get("grades", []) if 0 <= it["id"] < len(top)}
    cache[r["qid"]] = {"top": [int(x) for x in top], "grades": [g.get(n, 0) for n in range(len(top))]}
with ThreadPoolExecutor(10) as ex: list(ex.map(go, rows))
json.dump(cache, open(cache_f, "w"))
def ndcg(rk, g): return sum(1 / math.log2(i + 2) for i, s in enumerate(rk[:10]) if s in g) / sum(1 / math.log2(i + 2) for i in range(min(len(g), 10)))
res = collections.defaultdict(list); miss = []
for r in rows:
    g = r["gold"]; rk, _ = srank(r, base(r)); c = cache[r["qid"]]; top = c["top"]
    rr = sorted(range(len(top)), key=lambda n: (-c["grades"][n], n)); new = [top[n] for n in rr] + rk[K:]
    res["base"].append(ndcg(rk, g)); res["rerank"].append(ndcg(new, g)); res["all10"].append(g <= set(new[:10]))
    res["qt_" + r["qt"]].append(ndcg(new, g))
    pos = sorted(new.index(x) + 1 for x in g)
    if max(pos) > len(pos): miss.append((r["qt"][:8], D[r["qid"]]["question"][:70], pos, [c["grades"][top.index(x)] if x in top else None for x in g]))
print(f"model {MODEL} K={K}  base ndcg@10 {100*np.mean(res['base']):.1f}  ->  rerank {100*np.mean(res['rerank']):.1f}   all@10 {100*np.mean(res['all10']):.1f}")
for k in sorted(res):
    if k.startswith("qt_"): print(f"   {k[3:]:28s} {100*np.mean(res[k]):.1f}")
print("still imperfect:"); [print("  ", m) for m in miss]
print(dict(C.TOK), dict(C.STATS), "rerank latency median/p90", np.median(LAT) if LAT else None, np.percentile(LAT, 90) if LAT else None)
