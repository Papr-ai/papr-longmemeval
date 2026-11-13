"""Run A: schema top-5 sessions + OFFICIAL LongMemEval CoT answer prompt (json history, as in run_generation.py) + official judge. Reader gpt-4o-2024-08-06."""
import json, os, sys, collections, numpy as np, time, importlib.util
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
cli = OpenAI(api_key=os.environ.get("OPENAI_API_KEY_LONGMEMEVALONLY","unused")); TOK = collections.Counter()
spec = importlib.util.spec_from_file_location("ev", "evaluate_qa.py"); ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
D = {q["question_id"]: q for q in json.load(open("longmemeval_s_cleaned.json"))}; qpos = {q: i for i, q in enumerate(D)}
keys = json.load(open("turn_keys.json")); kidx = {k: i for i, k in enumerate(keys)}
SE = np.load("turns_s1all.npy", mmap_mode="r"); SQ = np.load("questions_s1all.npy").astype(np.float32)
QIDS = [json.loads(l)["qid"] for l in open("answers_claude-haiku-4-5_Zh1_k5_dev.jsonl")]
SET = os.environ.get("SET", "dev")
if os.environ.get("QFILE"): QIDS = json.load(open(os.environ["QFILE"]))
if SET == "rest": _d = set(QIDS); QIDS = [q for q in D if q not in _d]
if os.environ.get("TDATE"):
    from tdate import is_temporal as _it
    QIDS = [x for x in QIDS if _it(D[x]["question"])]
K = int(os.environ.get("K", "5")); READER = os.environ.get("READER", "gpt-4o-2024-08-06"); FMT = os.environ.get("FMT", "json")
TPL = ("I will give you several history chats between you and a user. Please answer the question based on the relevant chat history. Answer the question step by step: "
       "first extract all the relevant information, and then reason over the information to get the answer.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer (step by step):")
PROMPT_V = os.environ.get("PV", "")
if PROMPT_V == "2":
    TPL = TPL.replace("then reason over the information to get the answer.", "then reason over the information to get the answer. "
        "Rules: (1) End with ONE committed final answer on its own line starting 'Final answer:'. Do not hedge with alternatives or ask the user questions; if the evidence is ambiguous, pick the most likely reading. "
        "(2) For questions about the current state ('now', 'currently', 'these days'), use the most recent statement in the chats, including the user's latest estimate, plan or change, over older values. "
        "(3) For 'X days/weeks/months ago', first compute the target date from the Current Date, then pick the event whose date is closest to it. "
        "(4) For 'how many more / how much do I need', give the remaining amount if the chats give both the goal and the current progress.")
if PROMPT_V == "3":
    TPL = TPL.replace("then reason over the information to get the answer.", "then reason over the information to get the answer. "
        "If the question asks about the current state (it says 'now', 'currently' or 'these days'), use the user's most recent statement, including a latest estimate or a completed move. "
        "If it says 'X days/weeks/months ago', first compute that target date from the Current Date, then pick the event dated closest to it. "
        "If it asks how many more or how much is still needed, and the chats give both the goal and the current progress, answer with the remaining amount.")
if os.environ.get("TRUST"):
    TPL = TPL.replace("then reason over the information to get the answer.", "then reason over the information to get the answer. "
        "Treat the user's own statements as true. If a value changed over time, use the latest one unless the question asks for an earlier one. "
        "Give one committed answer, not a range, a hedge or a request for more details.")
if PROMPT_V == "P":
    TPL = TPL.replace("then reason over the information to get the answer.", "then reason over the information to get the answer. "
        "Before answering, do a premise check: list the specific things the question assumes (the item, role, event, person, place or amount it names) and, for each, quote the exact line in the chats that states it. "
        "A similar but different thing does not count (a different item, a different job title, a different event, or a number the user never gave). "
        "If any assumed thing has no direct support, answer that the information provided is not enough and say what is missing. Otherwise answer normally.")
JR = json.load(open("jev_route.json")) if PROMPT_V == "J" else {}
RULES = {"temporal": "This question is about time. First compute any target date (e.g. 'two weeks ago') from the Current Date, write down the date of each candidate event from the session dates or the user's own words, then pick the event or interval that matches. Answer at the granularity the question uses.",
         "current_state": "This question asks about the current state. Values may have changed across sessions: list each mentioned value with its session date, then answer with the most recent one, including the user's latest estimate or a completed change.",
         "assistant_recall": "This question asks what you (the assistant) said earlier. Find that exact earlier reply and quote the requested part verbatim instead of generating something new."}
REMAIN = "If the chats give both the goal and the current progress, answer with the amount still needed."
def tpl(qid):
    j = JR.get(qid) or {}; r = RULES.get(j.get("kind"), "") if (j.get("conf") or 0) >= 0.6 else ""
    if j.get("kind") == "current_state" and j.get("prev", 0) > 0.9: r = ""
    if j.get("remain", 0) > 0.8: r = (r + " " + REMAIN).strip()
    return TPL.replace("then reason over the information to get the answer.", "then reason over the information to get the answer. " + r) if r else TPL
SRR = json.load(open(os.environ["SRR"])) if os.environ.get("SRR") else None
def top(qid):
    q = D[qid]; ti, si, S = [], [], []
    for sid, ss, sd in zip(q["haystack_session_ids"], q["haystack_sessions"], q["haystack_dates"]):
        t = [kidx[f"{sid}|{i}"] for i in range(len(ss)) if f"{sid}|{i}" in kidx]
        if t: ti += t; si += [len(S)] * len(t); S.append((sd, ss))
    X = np.asarray(SE[ti], np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True); v = SQ[qpos[qid]] / np.linalg.norm(SQ[qpos[qid]])
    a = np.full(len(S), -1e9); np.maximum.at(a, np.array(si), X @ v); rk = [int(x) for x in np.argsort(-a)]
    if SRR: c = SRR[qid]; top = c["top"]; rk = [top[n] for n in sorted(range(len(top)), key=lambda n: (-c["grades"][n], n))] + [x for x in rk if x not in top]
    if SRR and os.environ.get("DYNT"):
        c = SRR[qid]; keep = [c["top"][n] for n in range(len(c["top"])) if c["grades"][n] >= float(os.environ["DYNT"])]
        kk = max(int(os.environ.get("DYNMIN", "2")), min(K, len(keep))); return [S[x] for x in rk[:kk]]
    if os.environ.get("TDATE"):
        from tdate import is_temporal
        if is_temporal(q["question"]): return [S[x] for x in rk[:int(os.environ.get("TK", "8"))]]
    return [S[x] for x in rk[:K]]
import urllib.request
def gcall(m, prompt):
    body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": 32000}}
    for a in range(5):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={os.environ['GOOGLE_API_KEY']}", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300))
            u = d.get("usageMetadata", {}); TOK[m + "_in"] += u.get("promptTokenCount", 0); TOK[m + "_out"] += u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
            return "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"]).strip()
        except Exception as e: err = e; time.sleep(3 + 5 * a)
    print("FAIL", str(err)[:100], flush=True); return ""
def ccall(m, prompt):
    import claude_oauth as co
    TH = int(os.environ.get("THINK", "8000"))
    body = {"model": m, "max_tokens": TH + 4000, "system": co.sysblk(None), "messages": [{"role": "user", "content": prompt}]}
    if TH: body["thinking"] = {"type": "enabled", "budget_tokens": TH}
    for a in range(4):
        try:
            d = co.raw(body); u = d.get("usage", {}); TOK[m + "_in"] += u.get("input_tokens", 0); TOK[m + "_out"] += u.get("output_tokens", 0)
            t = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text").strip()
            if t: return t
        except Exception as e: err = e; print("claude fail", str(e)[:150], flush=True); time.sleep(10 + 10 * a)
    print("FAIL claude", flush=True); return ""
def call(m, prompt, mt):
    if m.startswith("gemini"): return gcall(m, prompt)
    if m.startswith("claude"): return ccall(m, prompt)
    for a in range(5):
        try:
            kw = dict(max_completion_tokens=32000, **({"reasoning_effort": os.environ["REASON"]} if os.environ.get("REASON") and m == READER else {})) if m.startswith("gpt-5") else dict(temperature=0, max_tokens=mt)
            r = cli.chat.completions.create(model=m, messages=[{"role": "user", "content": prompt}], n=1, **kw)
            TOK[m + "_in"] += r.usage.prompt_tokens; TOK[m + "_out"] += r.usage.completion_tokens; return r.choices[0].message.content.strip()
        except Exception as e: err = e; time.sleep(3 + 5 * a)
    print("FAIL", str(err)[:100], flush=True); return ""
from tdate import pdate, target
def dl(sd, q):
    if not os.environ.get("DLBL"): return sd
    n = (pdate(q["question_date"]) - pdate(sd)).days; return f"{sd} ({n} days before the Current Date)"
def tnote(q):
    if not os.environ.get("DLBL"): return ""
    t, w = target(q["question"], q["question_date"])
    return f"\nNote: the time the question refers to is around {t.strftime('%Y/%m/%d')} (give or take {w} days)." if t else ""
import threading
_LK = threading.Lock(); PART = f"official_{os.environ.get('TAG','A')}_partial.jsonl"
def one(qid):
    q = D[qid]; ch = sorted(top(qid), key=lambda x: x[0]); hs = ""
    for i, (sd, ss) in enumerate(ch):
        s = ("\n" + json.dumps(ss)) if FMT == "json" else "".join("\n\n{}: {}".format(t["role"], t["content"].strip()) for t in ss)
        hs += "\n### Session {}:\nSession Date: {}\nSession Content:\n{}\n".format(i + 1, dl(sd, q), s)
    a = call(READER, tpl(qid).format(hs, q["question_date"] + tnote(q), q["question"]), 800)
    r = dict(qid=qid, qt=q["question_type"], ans=a, ok=None)
    with _LK: open(PART, "a").write(json.dumps(r) + "\n")
    return r
if os.environ.get("QN"): QIDS = QIDS[:int(os.environ["QN"])]
with ThreadPoolExecutor(int(os.environ.get("PAR", "8"))) as ex: R = list(ex.map(one, QIDS))
out = f"official_{os.environ.get('TAG', 'A')}{os.environ.get('REASON','')}{('pv'+PROMPT_V) if PROMPT_V else ''}{('dyn'+os.environ['DYNT']) if os.environ.get('DYNT') else ''}{('T'+os.environ.get('TK','8')) if os.environ.get('TDATE') else ''}_{READER}_{FMT}_k{K}_{SET}.jsonl"; open(out, "w").write("".join(json.dumps(r) + "\n" for r in R))
print(f"{READER} k{K} generated {len(R)} answers (NOT judged) -> {out}")
