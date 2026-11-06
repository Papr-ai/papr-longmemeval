"""Full re-embed: all 99,053 user turns (turn_keys.json order) + 500 questions with papr-embed-schema-1-0.6b under memory:personal:1.6.0.
Checkpointed in shards of 2048 turns under emb_s1_all/; reruns skip finished shards."""
import os, json, time, urllib.request, numpy as np
from concurrent.futures import ThreadPoolExecutor
MODEL = "papr-embed-schema-1-0.6b"; SID = "memory:personal:1.6.0"; OUT = "emb_s1_all"; os.makedirs(OUT, exist_ok=True)
H = {"X-API-Key": os.environ["PAPR_API_KEY"], "Content-Type": "application/json"}
I = "Given a question about the user's past conversations, retrieve the conversation turn that helps answer it"
def post(texts, q=False):
    b = {"model": MODEL, "input": texts, "input_type": "query" if q else "document", "band_schema_id": SID}
    if q: b["instruction"] = I
    for a in range(12):
        try: return [x["embedding"] for x in sorted(json.load(urllib.request.urlopen(urllib.request.Request("https://memory.papr.ai/v1/embeddings", json.dumps(b).encode(), H), timeout=300))["data"], key=lambda x: x["index"])]
        except Exception as e: print("retry", a, str(e)[:100], flush=True); time.sleep(min(60, 3 * (a + 1)))
    raise SystemExit("failed")
data = json.load(open("longmemeval_s_cleaned.json"))
txt = {f"{sid}|{i}": (t["content"].strip() or "(empty message)")[:6000] for q in data for sid, s in zip(q["haystack_session_ids"], q["haystack_sessions"]) for i, t in enumerate(s) if t["role"] == "user"}
keys = json.load(open("turn_keys.json")); S = 2048; t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    for si in range(0, len(keys), S):
        f = f"{OUT}/shard_{si // S:03d}.npy"
        if os.path.exists(f): continue
        ks = keys[si:si + S]; ch = [ks[i:i + 32] for i in range(0, len(ks), 32)]
        out = list(ex.map(lambda c: post([txt[k] for k in c]), ch))
        np.save(f, np.array([v for c in out for v in c], np.float16))
        print(f"shard {si // S + 1}/{(len(keys) + S - 1) // S} {time.time() - t0:.0f}s", flush=True)
qs = [q["question"] for q in data]
Q = [v for i in range(0, len(qs), 32) for v in post(qs[i:i + 32], True)]
np.save("questions_s1all.npy", np.array(Q, np.float32))
np.save("turns_s1all.npy", np.concatenate([np.load(f"{OUT}/shard_{i:03d}.npy") for i in range((len(keys) + S - 1) // S)]))
print("done", f"{time.time() - t0:.0f}s", flush=True)
