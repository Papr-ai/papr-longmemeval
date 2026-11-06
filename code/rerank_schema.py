"""Gemini session reranker (same prompt/grades as rerank40.py FINE=1) on top of SCHEMA embedding top-20. Logs per-question latency + tokens."""
import json, os, time, collections, numpy as np
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault("ROWS", "full_rows2.pkl"); os.environ["FINE"] = "1"
g = {"__name__": "rr"}; exec(open("rerank40.py").read().split("cache_f =")[0], g)
C, P, SCH, digest, D, MODEL = g["C"], g["P"], g["SCH"], g["digest"], g["D"], g["MODEL"]
qpos = {q: i for i, q in enumerate(D)}; keys = json.load(open("turn_keys.json")); kidx = {k: i for i, k in enumerate(keys)}
SE = np.load("turns_s1all.npy", mmap_mode="r"); SQ = np.load("questions_s1all.npy").astype(np.float32)
QIDS = [json.loads(l)["qid"] for l in open("answers_claude-haiku-4-5_Zh1_k5_dev.jsonl")]
if os.environ.get("RRSET") == "all": QIDS = list(D)
if os.environ.get("NQ"): QIDS = sorted(QIDS, key=lambda x: D[x]["question_type"] not in ("multi-session", "temporal-reasoning"))[:int(os.environ["NQ"])]
OUT = os.environ.get("RROUT", "rerank_schema_dev.json"); cache = json.load(open(OUT)) if os.path.exists(OUT) else {}
def go(qid):
    if qid in cache: return
    q = D[qid]; ti, si, S = [], [], []
    for sid, ss, sd in zip(q["haystack_session_ids"], q["haystack_sessions"], q["haystack_dates"]):
        t = [kidx[f"{sid}|{i}"] for i in range(len(ss)) if f"{sid}|{i}" in kidx]
        if t: ti += [(x, i) for x, i in zip(t, [int(k.split("|")[1]) for k in (keys[x] for x in t)])]; si += [len(S)] * len(t); S.append((sid, sd, ss))
    X = np.asarray(SE[[x for x, _ in ti]], np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True); v = SQ[qpos[qid]] / np.linalg.norm(SQ[qpos[qid]]); sc = X @ v
    a = np.full(len(S), -1e9); np.maximum.at(a, np.array(si), sc); top = [int(x) for x in np.argsort(-a)[:20]]
    bys = collections.defaultdict(dict)
    for (x, i), s, n in zip(ti, sc, si): bys[n][i] = float(s)
    body = "\n\n".join(f"[{n}]\n" + digest(S[k][0], S[k][1], S[k][2], bys[k]) for n, k in enumerate(top))
    t0 = time.time(); i0, o0 = C.TOK[MODEL + "_in"], C.TOK[MODEL + "_out"]
    out = C.gem(P.format(qd=q["question_date"], q=q["question"], s=body), SCH, model=MODEL)
    gr = {it["id"]: it["grade"] for it in (out or {}).get("grades", []) if 0 <= it["id"] < len(top)}
    cache[qid] = {"top": top, "grades": [gr.get(n, 0) for n in range(len(top))], "lat": round(time.time() - t0, 3)}
with ThreadPoolExecutor(int(os.environ.get("RRPAR", "10"))) as ex: list(ex.map(go, QIDS))
json.dump(cache, open(OUT, "w"))
L = [c["lat"] for c in cache.values() if "lat" in c]; ti_, to_ = C.TOK[MODEL + "_in"], C.TOK[MODEL + "_out"]
st = dict(model=MODEL, n=len(cache), lat_median=float(np.median(L)), lat_p90=float(np.percentile(L, 90)), tok_in=ti_, tok_out=to_, cost_usd=round(ti_ * 0.30e-6 + to_ * 2.50e-6, 3))
json.dump(st, open(OUT.replace(".json", "_stats.json"), "w")); print("RERANK", st, flush=True)
