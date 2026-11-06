"""Inference-time features from an ItemView (no gold/annotation fields allowed)."""
import re, math, hashlib, datetime as dt
FORBIDDEN = {"final", "annotations", "gold", "gold_value", "label"}
HEDGE = re.compile(r"(~|≈|\?|\best\.?\b|\bapprox|\btbd\b|\bpending\b|\bprelim|\bunverified\b|\bdraft\b)", re.I)
ERRATUM = re.compile(r"\b(erratum|correction|corrected|supersedes|retract)", re.I)
DATE = re.compile(r"(20\d{2}|19\d{2})-(\d{2})-(\d{2})")
NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")
def assert_view(item):
    bad = FORBIDDEN & set(item); assert not bad, f"gold fields leaked into pipeline: {bad}"
def origin(ev):
    o = ev.get("source", {}).get("origin_id")
    return o or hashlib.sha256(re.sub(r"\s+", " ", ev["text"].lower()).encode()).hexdigest()[:16]
def _date(t):
    m = DATE.search(t); return dt.date(*map(int, m.groups())) if m else None
def _num(t):
    m = NUM.findall(t); return float(m[-1]) if m else None
def pair_features(a, b, depth=None, ctx_sim=0.0, agree_origins=1, disagree_origins=0):
    ra = HEDGE.search(a["text"]) or HEDGE.search(b["text"]); er = ERRATUM.search(a["text"] + " " + b["text"])
    f1 = (-1.0 if ra else 0.0) + (0.5 if er else 0.0); f1_miss = 0.0
    va, vb = _num(a["text"]), _num(b["text"])
    f2 = abs(va - vb) / max(abs(va), abs(vb), 1e-9) if va is not None and vb is not None else 0.0
    da, db = _date(a["text"]), _date(b["text"])
    f3, f3_miss = (math.log1p(abs((da - db).days)), 0.0) if da and db else (0.0, 1.0)
    f4, f4_miss = (float(depth), 0.0) if depth is not None else (0.0, 1.0)
    slots = [va is not None, vb is not None, da is not None, db is not None, bool(a.get("source")), bool(b.get("source"))]
    f6 = sum(slots) / len(slots)
    return [f1, f2, f3, f4, ctx_sim, f6, float(agree_origins), float(disagree_origins), f1_miss, f3_miss, f4_miss]
def dedup_count(evs):
    return len({origin(e) for e in evs})
