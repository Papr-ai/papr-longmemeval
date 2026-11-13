import json, os, importlib.util, concurrent.futures as cf, threading, time, collections
from openai import OpenAI
s=importlib.util.spec_from_file_location("ev","evaluate_qa.py"); ev=importlib.util.module_from_spec(s); s.loader.exec_module(ev)
D={q["question_id"]:q for q in json.load(open("longmemeval_s_cleaned.json"))}
cli=OpenAI(api_key=os.environ["JK"])
A={}
fs=[f"official_K20b{n}_claude-opus-4-6_json_k20_dev.jsonl" for n in range(1,6)]+["official_K20miss_claude-opus-4-6_json_k20_dev.jsonl","official_K20s40_claude-opus-4-6_json_k20_dev.jsonl"]
for f in fs:
    for l in open(f):
        r=json.loads(l); A[r["qid"]]=r["ans"]
print("answers",len(A),flush=True)
OUT="k20_gpt4o_judged.jsonl"; done={}
if os.path.exists(OUT):
    for l in open(OUT): r=json.loads(l); done[r["qid"]]=r
LK=threading.Lock()
def one(qid):
    if qid in done: return done[qid]
    q=D[qid]; p=ev.get_anscheck_prompt(q["question_type"],q["question"],q["answer"],A[qid],abstention="_abs" in qid)
    for a in range(5):
        try:
            c=cli.chat.completions.create(model="gpt-4o-2024-08-06",messages=[{"role":"user","content":p}],n=1,temperature=0,max_tokens=10)
            j=c.choices[0].message.content.strip(); break
        except Exception as e: j=None; time.sleep(3*(a+1))
    r=dict(qid=qid,qt=q["question_type"],judge=j,ok=(None if j is None else "yes" in j.lower()))
    with LK: open(OUT,"a").write(json.dumps(r)+"\n")
    return r
with cf.ThreadPoolExecutor(8) as ex: R=list(ex.map(one,list(A)))
H={}
for l in open("../papr-longmemeval/results/papr_high_opus-4-6.jsonl.eval-results-gpt-4o"):
    r=json.loads(l); H[r["question_id"]]=r["autoeval_label"]["label"]
ok=sum(r["ok"] is True for r in R); print("K20 gpt-4o judged:",ok,"/",len(R),f"= {ok/len(R):.1%}","| None:",sum(r["ok"] is None for r in R))
print("High (gpt-4o):",sum(H.values()),"/",len(H))
by=collections.defaultdict(lambda:[0,0,0])
for r in R:
    b=by[r["qt"]]; b[0]+=r["ok"] is True; b[1]+=1; b[2]+=H[r["qid"]] is True
for t,b in sorted(by.items()): print(f"  {t:28s} K20 {b[0]}/{b[1]}  High {b[2]}/{b[1]}")
ab=[r for r in R if "_abs" in r["qid"]]; print("abstention K20",sum(r["ok"] is True for r in ab),"/",len(ab),"High",sum(H[r["qid"]] is True for r in ab))
print("GAINS (K20 right, High wrong):",sorted(r["qid"] for r in R if r["ok"] is True and H[r["qid"]] is not True))
print("LOSSES (High right, K20 wrong):",sorted(r["qid"] for r in R if r["ok"] is not True and H[r["qid"]] is True))
print("STILL WRONG:",sorted(r["qid"] for r in R if r["ok"] is not True and H[r["qid"]] is not True))
