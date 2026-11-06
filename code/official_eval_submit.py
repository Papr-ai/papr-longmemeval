"""Run the UNMODIFIED official LongMemEval evaluate_qa.py (gpt-4o judge) on both submission files, in parallel."""
import os, subprocess, sys
env = dict(os.environ, OPENAI_API_KEY=os.environ["OPENAI_API_KEY_LONGMEMEVALONLY"])
ev = os.path.abspath("../LongMemEval/src/evaluation"); ref = os.path.abspath("longmemeval_s_cleaned.json")
files = [os.path.abspath(f"submit/{n}.jsonl") for n in os.environ.get("NAMES", "papr_high_opus-4-6,papr_low_gpt-5-mini").split(",")]
ps = [subprocess.Popen([sys.executable, "evaluate_qa.py", "gpt-4o", f, ref], cwd=ev, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for f in files]
for f, p in zip(files, ps):
    out = p.communicate()[0]
    print("=====", os.path.basename(f), "exit", p.returncode); print("\n".join(l for l in out.splitlines() if "Accuracy" in l or ":" in l and len(l) < 120)[-3000:])
for f in files:
    r = subprocess.run([sys.executable, "print_qa_metrics.py", f + ".eval-results-gpt-4o", ref], cwd=ev, env=env, capture_output=True, text=True)
    print("=====", os.path.basename(f), "print_qa_metrics\n", r.stdout[-2000:], r.stderr[-500:], flush=True)
