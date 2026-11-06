import re
"""Schema v0.2 coverage test on ORACLE sessions (only evidence sessions) for the 40 small-test questions.
1) extract v0.2 records per oracle session (+ question-side plan)
2) judge (gemini-2.5-flash): given ONLY the records, can the gold answer be derived? for v0.1 and v0.2
"""
import json, os, sys, time, urllib.request, collections
from concurrent.futures import ThreadPoolExecutor
KEY = os.environ["GOOGLE_API_KEY"]
def url(m): return f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={KEY}"
TOK = collections.Counter()
STATS = collections.Counter()
def _salvage(txt):
    try: return json.loads(txt)
    except Exception: pass
    i = len(txt)
    for _ in range(300):
        i = txt.rfind("}", 0, i)
        if i < 0: return None
        for tail in ("]}", "}"):
            try: return json.loads(txt[:i + 1] + tail)
            except Exception: pass
    return None
def gem(prompt, schema, model="gemini-2.5-flash-lite", max_out=16000):
    err = None
    if model.startswith("groq:"):
        for a in range(4):
            try:
                body = {"model": model[5:], "messages": [{"role": "user", "content": prompt + '\n\nReturn only JSON: {"grades": [{"id": <int>, "grade": <int>}, ...]} covering every session id.'}],
                        "response_format": {"type": "json_object"}, "temperature": 0, "max_completion_tokens": 2000, "reasoning_effort": os.environ.get("GROQ_EFFORT", "low")}
                d = json.load(urllib.request.urlopen(urllib.request.Request("https://api.groq.com/openai/v1/chat/completions", json.dumps(body).encode(),
                        {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["GROQ_API_KEY"], "User-Agent": "lme"}), timeout=60))
                u = d.get("usage", {}); TOK[model + "_in"] += u.get("prompt_tokens", 0); TOK[model + "_out"] += u.get("completion_tokens", 0)
                return json.loads(d["choices"][0]["message"]["content"])
            except Exception as ex: err = ex; time.sleep(1 + 2 * a)
        print("FAIL", str(err)[:200], flush=True); return None
    for a in range(3):
        body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json",
                "responseSchema": schema, "temperature": 0 if a == 0 else 0.4, "maxOutputTokens": max_out}}
        if os.environ.get("THINK") is not None and "2.5-flash" in model and "lite" not in model: body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": int(os.environ["THINK"])}
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(url(model), json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=int(os.environ.get("GEM_TIMEOUT", "90"))))
            u = d.get("usageMetadata", {}); TOK[model + "_in"] += u.get("promptTokenCount", 0); TOK[model + "_out"] += u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
            c = d["candidates"][0]; r = _salvage(c["content"]["parts"][0]["text"])
            if c.get("finishReason") == "MAX_TOKENS": STATS["truncated"] += 1
            if r is not None: return r
            err = "unparseable"; STATS["unparseable"] += 1
        except Exception as ex: err = ex; time.sleep(2 + 3 * a)
    STATS["failed"] += 1; print("FAIL", str(err)[:120], flush=True); return None
EVENT_CLASSES = ["attended_event", "attended_wedding", "attended_concert_show", "attended_class_workshop", "travel_trip", "visited_place",
  "purchase", "sold_item", "received_gift", "gave_gift", "joined_group", "left_group", "started_activity", "stopped_activity",
  "completed_achievement", "personal_record", "health_issue", "medical_appointment", "work_job_change", "work_project", "moved_home",
  "relationship_family_event", "pet_event", "hobby_session", "fitness_session", "reading_media", "watched_media", "cooking_food",
  "finance_transaction", "volunteering_charity", "social_post", "other"]
S = {"type": "STRING"}; SN = {"type": "STRING", "nullable": True}; NN = {"type": "NUMBER", "nullable": True}
REC = {"type": "OBJECT", "properties": {
  "kind": {"type": "STRING", "enum": ["state", "event", "preference"]},
  "subject": S, "aliases": {"type": "ARRAY", "items": S}, "entity_type": S, "attribute": S,
  "value": S, "value_type": {"type": "STRING", "enum": ["number", "duration", "date", "money", "text", "list"]}, "unit": SN,
  "slot_key": S, "supersedes_prior": {"type": "BOOLEAN"},
  "event_class": {"type": "STRING", "enum": EVENT_CLASSES}, "event": SN,
  "event_date": SN, "date_precision": {"type": "STRING", "enum": ["day", "week", "month", "year", "unknown"]},
  "date_basis": {"type": "STRING", "enum": ["explicit", "relative_resolved", "session_date", "unknown"]}, "relative_expr": SN,
  "start_date": SN, "quantity": NN, "amount": NN, "currency": SN, "people": {"type": "ARRAY", "items": S}, "place": SN,
  "duration_value": NN, "duration_unit": {"type": "STRING", "enum": ["day", "week", "month", "year", "none"]}, "duration_as_of": SN, "stance": SN, "source_turn": {"type": "INTEGER"}, "evidence": S},
  "required": ["kind", "subject", "attribute", "value", "value_type", "slot_key", "supersedes_prior", "event_class", "date_precision", "date_basis", "source_turn", "evidence"]}
DOC = {"type": "OBJECT", "properties": {"records": {"type": "ARRAY", "items": REC}}, "required": ["records"]}
DOC_PROMPT = """You build a personal memory store from a chat. Extract EVERY fact the USER reveals about themselves or their life
(states, events, possessions, people, numbers, plans, preferences), including facts mentioned in passing inside a request for help.
Rules:
- kind: state (a current property: job, PB time, pet name, count owned), event (something that happened/will happen at a time), preference.
- subject: lowercase canonical noun phrase without 'my' (e.g. '5k personal best', 'dog max', 'book club page turners'); aliases: other surface forms used.
- slot_key: '<subject>|<attribute>' lowercase; the SAME real-world property must always get the same slot_key so newer values replace older ones.
- supersedes_prior: true if the user indicates a change/new value ('now', 'new record', 'switched', 'upgraded', 'moved').
- event_class: pick the closest from the enum; event: short description.
- event_date: ISO YYYY-MM-DD; resolve relative expressions ('last Saturday', 'two weeks ago', 'yesterday') against the session date; if no time cue,
  use the session date with date_basis=session_date. relative_expr: the raw phrase. start_date: when a state/membership/habit began, if given or derivable.
- quantity: counts (3 games, 2 tickets); amount+currency: money. Separate each instance of a countable event into its own record.
- source_turn: index of the user turn. evidence: short verbatim quote.
- Several records per turn are expected. Facts mentioned in passing inside a request for help MUST be extracted
  (e.g. 'I had bronchitis I initially thought was a cold', 'I missed the fun run due to work').
- Keep qualifiers that change meaning in value/evidence: 'each way', 'per week', 'initially thought X, turned out Y', 'due to <reason>',
  the place/trip a number belongs to. For belief-vs-reality, write attribute='initially believed' and attribute='actual' as two records.
- Durations/tenures (role, membership, habit, ownership, how long something took): put the number in duration_value and the unit in
  duration_unit exactly as stated ('3 years and 9 months' -> 3.75 year; '2 months' -> 2 month). Do NOT compute start dates yourself;
  set duration_as_of only if the duration is measured to a date other than the session date. start_date only for an explicitly
  stated start ('I started in March 2022'). If a role changed, record an event for the change with its date.
- event_date is REQUIRED for every event: use the stated/relative date; if none, use the session date with date_basis=session_date.
  'just finished' / 'recently' = session date (date_precision=week).
- Each countable instance (a run, a tank, a trip, a purchase, a book) is its own record with its own date, quantity and qualifiers
  (who it was for, why it was missed, where).
Session date: <<DATE>>

<<TURNS>>"""
GLEAN_PROMPT = """Below are a user's turns from one chat and the memory records already extracted from them.
Return ONLY additional records that were missed: facts mentioned in passing, extra instances of countable things (each separately),
reasons/qualifiers (who for, why, where, each way), belief-vs-reality pairs, durations (duration_value/unit), and dates for events.
Do not repeat records already present. Same rules and conventions as before:
""" + DOC_PROMPT.split("Session date:")[0] + """
Session date: <<DATE>>

USER TURNS:
<<TURNS>>

ALREADY EXTRACTED:
<<RECS>>"""
QP = {"type": "OBJECT", "properties": {
  "operation": {"type": "STRING", "enum": ["lookup", "latest_value", "count", "sum", "order", "date_diff", "duration", "list", "preference"]},
  "target_slots": {"type": "ARRAY", "items": S}, "target_event_classes": {"type": "ARRAY", "items": {"type": "STRING", "enum": EVENT_CLASSES}},
  "target_subjects": {"type": "ARRAY", "items": S}, "window_from": SN, "window_to": SN, "n_events_needed": {"type": "INTEGER"}},
  "required": ["operation", "target_slots", "target_event_classes", "target_subjects", "n_events_needed"]}
Q_PROMPT = """A user asks about their own past chats. Do NOT answer. Produce a retrieval plan in the same conventions as the memory store:
slot_key '<subject>|<attribute>' lowercase, subjects without 'my', event_class from the enum. Resolve relative time to an ISO window using
the question date. n_events_needed: how many distinct records must be found (e.g. 2 for 'which came first', N for counting).
Question date: <<DATE>>
Question: <<Q>>"""
JUDGE = {"type": "OBJECT", "properties": {"derived_answer": S, "verdict": {"type": "STRING", "enum": ["yes", "partial", "no"]}, "missing": S}, "required": ["derived_answer", "verdict", "missing"]}
J_PROMPT = """You are auditing a memory extraction schema. Using ONLY the extracted records below (not outside knowledge, not the original chat),
try to answer the question. Then compare with the gold answer.
verdict: yes = records contain everything needed and your answer matches gold; partial = some needed info present but something missing/wrong
(e.g. one of two events, a date missing, an instance missing for a count); no = needed info absent.
missing: what specific information the records lack (empty if yes).
Question date: <<QD>>
Question: <<Q>>
Gold answer: <<A>>
Records:
<<R>>"""

import datetime as _dt
UNIT_DAYS = {"day": 1, "week": 7, "month": 30.44, "year": 365.25}
def pdate(x):
    if not x: return None
    x = str(x).replace("/", "-")
    try: return _dt.date.fromisoformat(x[:10])
    except Exception: pass
    m = re.match(r"(\d{4})-(\d{2})$", x[:7]) if len(x) >= 7 else None
    try: return _dt.date(int(m.group(1)), int(m.group(2)), 15) if m else None
    except Exception: return None
def post(recs, session_date):
    """Deterministic post-processing: dedupe, start_date from duration (code, not LLM), session-date fill for undated events."""
    sd = pdate(session_date[:10]) or _dt.date(2023, 1, 1); seen = set(); out = []
    for r in recs:
        k = (str(r.get("slot_key", "")).lower().strip(), str(r.get("value", "")).lower().strip(), r.get("event_date"), r.get("source_turn"))
        if k in seen: continue
        seen.add(k)
        dv, du = r.get("duration_value"), r.get("duration_unit")
        if dv and du in UNIT_DAYS:
            asof = pdate(r.get("duration_as_of")) or sd
            try:
                if 0 < float(dv) < 200 * (365.25 / UNIT_DAYS[du]):
                    r["start_date"] = (asof - _dt.timedelta(days=round(float(dv) * UNIT_DAYS[du]))).isoformat(); r["start_basis"] = "computed_from_duration"
            except Exception: pass
        ed = pdate(r.get("event_date"))
        if r.get("kind") == "event":
            if ed: r["event_date"] = ed.isoformat()
            else: r["event_date"] = sd.isoformat(); r["date_basis"] = "session_date"
        out.append(r)
    return out
