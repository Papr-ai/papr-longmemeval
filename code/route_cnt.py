"""Route: extraction (count3) answers only for true count/total questions; date/duration questions keep plain Opus answer.
Rule fixed before scoring: DUR = how many <time unit> | how long | ago."""
import json,re
DUR=re.compile(r'\bhow many (days|weeks|months|years|hours|minutes)\b|\bhow long\b|\bago\b',re.I)
D={q['question_id']:q for q in json.load(open('longmemeval_s_cleaned.json'))}
tot=0
for s in ('dev','rest'):
  B={r['qid']:r for r in map(json.loads,open(f'official_RRF_claude-opus-4-6_json_k5_{s}.jsonl'))}
  E={r['qid']:r for r in map(json.loads,open(f'official_CNT3opus_claude-opus-4-6_k5_{s}.jsonl'))}
  ok={q:r['ok'] for q,r in B.items()}; routed=0; fx=[];br=[];skipped=[]
  for q,r in E.items():
    if DUR.search(D[q]['question']):
      if r['ok']!=B[q]['ok']: skipped.append(('fix' if r['ok'] else 'break',D[q]['question'][:70]))
      continue
    routed+=1; ok[q]=r['ok']
    if r['ok'] and not B[q]['ok']: fx.append(D[q]['question'][:70])
    if B[q]['ok'] and not r['ok']: br.append(D[q]['question'][:70])
  n=sum(ok.values()); tot+=n
  print(f'{s}: routed {routed}/{len(E)} to extraction -> {n}/{len(B)}');[print('  FIXED',x) for x in fx];[print('  BROKE',x) for x in br];[print('  (kept plain Opus, would have been',k,')',x) for k,x in skipped]
print('TOTAL',tot,'/500 =',round(tot/5,1),'%')
