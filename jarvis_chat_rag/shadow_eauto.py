"""Shadow-mode adapter: frozen NEURON-4 E-auto as a JARVIS chat sidecar.

Hard rules:
- Import neuron4 run_neuron4 / n3lib read-only. Never write under neuron4/.
- Do NOT replace the primary chat answer. This module only produces a sidecar.
- Build memories from raw fact texts (auto_extract spirit). No gold labels as features.
- Native N4 extract_facts_from_text keeps NUMERIC table cells only. Open chat facts are
  mostly strings, so this adapter ALSO extracts string cells from the same markdown
  table format (documented limitation — not a fluency fake).

Usage:
  from shadow_eauto import ShadowEAuto
  se = ShadowEAuto.from_knowledge()
  print(se.answer_sidecar("What is the capital of Zorblax?"))
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
KNOW = HERE / "knowledge"
UPLOAD_JSONL = KNOW / "uploaded_facts.jsonl"
NOTES_JSONL = KNOW / "jarvis_notes.jsonl"
N4_ROOT = Path("/workspace/jarvis/neuron/neuron4")

# Import frozen NEURON-4 (read-only). Path insert only; never write there.
sys.path.insert(0, str(N4_ROOT))
import run_neuron4 as N4  # noqa: E402
import n3lib as N  # noqa: E402

PROJECT = "chat_rag/shadow"
DATE = "2026-11-06"
UNC = N.UNC  # same uncertainty markers as N4

LIMITATION = (
    "E-auto Graph expects structured {project, entity, attr, intent} queries and "
    "entity/attr/value memories. Native N4 auto_extract only keeps NUMERIC markdown "
    "table cells. This adapter (1) renders each fact card as a 2-column markdown table "
    "plus the raw sentence, (2) runs N4.auto_extract for numeric cells, (3) adds a "
    "string-cell extractor over the SAME tables so open chat values work, (4) maps NL "
    "questions to entity/attr by token overlap — not fluent free-form QA. Uncertain "
    "flags come from UNC regex markers only (no gold-trained classifier)."
)


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


# ---------- fact → table rendering (adapter preprocessing) ----------
_PATTERNS = [
    # The capital of X is Y
    (re.compile(r"(?i)^(?:the\s+)?capital of (.+?) is (.+?)\.?$"),
     lambda m: (m.group(1), "capital", m.group(2))),
    # X was written by Y / X wrote Y
    (re.compile(r"(?i)^(.+?) was written by (.+?)\.?$"),
     lambda m: (m.group(1), "author", m.group(2))),
    (re.compile(r"(?i)^(.+?) wrote (.+?)\.?$"),
     lambda m: (m.group(2), "author", m.group(1))),
    # X discovered / invented / founded Y  OR  Y was discovered by X
    (re.compile(r"(?i)^(.+?) discovered (.+?)\.?$"),
     lambda m: (m.group(2), "discoverer", m.group(1))),
    (re.compile(r"(?i)^(.+?) was discovered by (.+?)\.?$"),
     lambda m: (m.group(1), "discoverer", m.group(2))),
    (re.compile(r"(?i)^(.+?) invented (.+?)\.?$"),
     lambda m: (m.group(2), "inventor", m.group(1))),
    (re.compile(r"(?i)^(.+?) was invented by (.+?)\.?$"),
     lambda m: (m.group(1), "inventor", m.group(2))),
    (re.compile(r"(?i)^(.+?) (?:was )?founded by (.+?)\.?$"),
     lambda m: (m.group(1), "founder", m.group(2))),
    (re.compile(r"(?i)^(.+?) founded (.+?)\.?$"),
     lambda m: (m.group(2), "founder", m.group(1))),
    # numeric-ish
    (re.compile(r"(?i)^(?:the\s+)?(?:element\s+)?(.+?) has atomic number (.+?)\.?$"),
     lambda m: (m.group(1), "atomic number", m.group(2))),
    (re.compile(r"(?i)^(.+?) launched in (.+?)\.?$"),
     lambda m: (m.group(1), "launch year", m.group(2))),
    (re.compile(r"(?i)^(.+?) (?:began|started|was fought) in (.+?)\.?$"),
     lambda m: (m.group(1), "year", m.group(2))),
    (re.compile(r"(?i)^(?:world war i|the first world war) began in (.+?)\.?$"),
     lambda m: ("world war i", "year", m.group(1))),
    (re.compile(r"(?i)^(.+?) has (.+?) moons?\.?$"),
     lambda m: (m.group(1), "moons", m.group(2))),
    (re.compile(r"(?i)^(?:planet\s+)?(.+?) has (.+?) moons?\.?$"),
     lambda m: (m.group(1), "moons", m.group(2))),
    (re.compile(r"(?i)^(.+?) is (.+?) in color\.?$"),
     lambda m: (m.group(1), "color", m.group(2))),
    (re.compile(r"(?i)^(?:the\s+)?(?:ceo of|motto of|home port of|largest city on|goal of)\s+(.+?) is (.+?)\.?$"),
     lambda m: (m.group(1), "value", m.group(2))),
    (re.compile(r"(?i)^(?:in\s+)?(.+?), the word for (.+?) is (.+?)\.?$"),
     lambda m: (m.group(2), f"word in {m.group(1)}", m.group(3))),
    (re.compile(r"(?i)^(?:in\s+)?(.+?), a ring scores (.+?) points?\.?$"),
     lambda m: ("ring", f"points in {m.group(1)}", m.group(2))),
    (re.compile(r"(?i)^(.+?) uses (.+?)\.?$"),
     lambda m: (m.group(1), "uses", m.group(2))),
    (re.compile(r"(?i)^(.+?) lives in (.+?)\.?$"),
     lambda m: (m.group(1), "habitat", m.group(2))),
    (re.compile(r"(?i)^(?:the\s+)?(?:river\s+)?(.+?) flows into (.+?)\.?$"),
     lambda m: (m.group(1), "flows into", m.group(2))),
    (re.compile(r"(?i)^(.+?)'s famous work is (.+?)\.?$"),
     lambda m: (m.group(1), "famous work", m.group(2))),
    (re.compile(r"(?i)^(?:the\s+)?festival of (.+?) is held in (.+?)\.?$"),
     lambda m: (m.group(1), "month", m.group(2))),
    # research metric cards: "X was Y" / "X is Y" with numeric or short value
    (re.compile(
        r"(?i)^(?:NEURON-4 E-auto confident-wrong rate was|NEURON-4 E-auto confident-wrong rate is)\s+(.+?)(?:\s+on\b.*)?$"
     ), lambda m: ("neuron-4 e-auto", "confident-wrong", m.group(1).rstrip("."))),
    (re.compile(r"(?i)^(?:training-free chat_rag full eval hybrid accuracy was)\s+(.+?)(?:\s+on\b.*)?$"),
     lambda m: ("chat_rag hybrid", "accuracy", m.group(1).rstrip("."))),
    (re.compile(r"(?i)^(?:training-free chat_rag generation accuracy was)\s+(.+?)(?:\s+on\b.*)?$"),
     lambda m: ("chat_rag gen", "accuracy", m.group(1).rstrip("."))),
    (re.compile(r"(?i)^(?:training-free chat_rag CAD accuracy was)\s+(.+?)(?:\s+on\b.*)?$"),
     lambda m: ("chat_rag cad", "accuracy", m.group(1).rstrip("."))),
    (re.compile(r"(?i)^(?:training-free chat_rag pack-routed retrieval accuracy was)\s+(.+?)(?:\s+on\b.*)?$"),
     lambda m: ("chat_rag routed retrieval", "accuracy", m.group(1).rstrip("."))),
    (re.compile(r"(?i)^shadow probe metric (\w+) is (.+?)\.?$"),
     lambda m: (f"shadow probe {m.group(1).lower()}", "value", m.group(2).strip().rstrip("."))),
    (re.compile(
        r"(?i)^in the engine_v3 STP ablation, (STP) was the only arm that beat BASE"
     ), lambda m: ("engine_v3 ablation", "best arm vs base", m.group(1))),
    (re.compile(
        r"(?i)^the tutor pilot was (NOT SUPPORTED).*"
     ), lambda m: ("tutor pilot", "verdict", m.group(1))),
    (re.compile(
        r"(?i)^NEURON-5 P_energy did not beat E-auto on confident-wrong.*"
     ), lambda m: ("neuron-5", "beat e-auto on confident-wrong", "no")),
    (re.compile(
        r"(?i)^QMS-1 found defaults already best.*"
     ), lambda m: ("qms-1", "hardware case", "no")),
    (re.compile(
        r"(?i)^Hypothalamus A/B/C tests on 2026-11-06 were not supported.*"
     ), lambda m: ("hypothalamus abc", "supported", "no")),
    (re.compile(
        r"(?i)^callosum_scaled primary result was (PARTIAL).*"
     ), lambda m: ("callosum_scaled", "primary result", m.group(1))),
    (re.compile(
        r"(?i)^(?:the\s+)?real callosum map did not beat degree-matched shuffle.*"
     ), lambda m: ("callosum_scaled", "beat degree shuffle on transfer", "no")),
    (re.compile(
        r"(?i)^(?:On NEURON-4 corpus1, )?E-auto confident-wrong [\d\.]+ beat exact.*"
     ), lambda m: ("neuron-4 e-auto", "beat exact on confident-wrong", "yes")),
    (re.compile(
        r"(?i)^NEURON-4 evaluation labels are text-pattern rules, not human judgment\.?$"
     ), lambda m: ("neuron-4 labels", "human judgment", "no")),
]


def parse_triple(statement: str) -> Optional[Tuple[str, str, str]]:
    s = (statement or "").strip()
    if not s:
        return None
    # take first sentence
    first = re.split(r"(?<=[.!?])\s+", s, maxsplit=1)[0].strip()
    for rx, fn in _PATTERNS:
        m = rx.match(first)
        if m:
            try:
                ent, attr, val = fn(m)
            except Exception:
                continue
            ent = N.clean(ent).lower()
            attr = N.clean(attr).lower()
            val = N.clean(val).rstrip(".")
            if ent and attr and val:
                return ent, attr, val
    return None


def render_memory(row: dict, idx: int) -> dict:
    """Build an N4-style memory view from a fact card. Raw text + table for extract."""
    statement = (row.get("fact_statement") or row.get("text") or "").strip()
    # Prefer fact_statement; else first sentence of text
    if not row.get("fact_statement"):
        statement = re.split(r"(?<=[.!?])\s+", statement)[0].strip()
    rid = str(row.get("id") or f"row_{idx}")
    fname = f"{rid}.md"
    mid = f"/shadow/{fname}#0"
    triple = parse_triple(statement)
    body_parts = [f"## {rid}", statement, statement]
    if triple:
        ent, attr, val = triple
        # markdown table — numeric cells feed N4.auto_extract; all cells feed string extract
        # Preserve uncertainty markers from the statement in the value cell so E-auto can abstain
        cell_val = val
        if UNC.search(statement) and not UNC.search(cell_val):
            cell_val = f"{val} (approx, unverified)"
        body_parts += [
            "",
            f"| entity | {attr} |",
            "| --- | --- |",
            f"| {ent} | {cell_val} |",
        ]
    # Keep raw card text for BM25-ish token retrieval inside Graph.inv
    raw_extra = (row.get("text") or "")[:800]
    if raw_extra and raw_extra not in "\n".join(body_parts):
        body_parts += ["", raw_extra]
    body = "\n".join(body_parts)
    text = f"[meta date={DATE} project={PROJECT} file={fname}]\n{body}"
    return {
        "id": mid,
        "text": text,
        "project": PROJECT,
        "path": f"/shadow/{fname}",
        "_triple": triple,
        "_source_row": row,
    }


def extract_string_facts_from_tables(text: str, path: str, date: str, project: str) -> List[dict]:
    """Same table walk as N.extract_facts_from_text, but keep non-numeric string cells.

    Native N4 drops non-numeric cells (CELL_NUM). Chat needs strings. Documented.
    """
    facts = []
    prereg = Path(path).name.startswith("PREREG")
    for li, hdr, row, heading in N.parse_tables(text):
        if len(row) < 2 or len(hdr) < 2:
            continue
        ent = N.clean(row[0]).lower()
        if not ent or len(ent) > 80 or not re.search(r"[a-z]", ent):
            continue
        for k in range(1, min(len(hdr), len(row))):
            attr = N.clean(hdr[k]).lower()
            if not attr or len(attr) > 60 or not re.search(r"[a-z]", attr):
                continue
            cell = row[k]
            # Prefer numeric norm when present (match N4), else raw string
            m = N.CELL_NUM.search(cell)
            if m and N.norm_num(m.group(0)) is not None:
                v = N.norm_num(m.group(0))
            else:
                v = N.clean(cell)
                if not v or v.lower() in {"---", "nan", "none"}:
                    continue
            unc = bool(UNC.search(cell)) or bool(UNC.search(text[:500])) or prereg
            facts.append(
                dict(
                    entity=ent,
                    attr=attr,
                    value=v,
                    raw=cell.strip(),
                    path=str(path),
                    date=date,
                    project=project,
                    heading=heading.lower(),
                    line=li,
                    uncertain=unc,
                    prereg=False,  # never use gold prereg
                )
            )
    return facts


def build_facts(views: List[dict]) -> List[dict]:
    """N4.auto_extract (numeric) ∪ string-table extract. Drop gold fields."""
    # N4.auto_extract expects views with meta header; strips GOLD_F
    numeric = N4.auto_extract(views)
    # string path
    str_facts = []
    for m in views:
        mt = N4.HDR.match(m["text"])
        if not mt:
            continue
        date, proj, fname = mt.group(1), mt.group(2), mt.group(3)
        body = m["text"][mt.end():]
        for f in extract_string_facts_from_tables(body, "/x/" + fname, date, proj):
            f["path"] = m["path"]
            f["line"] = (m["id"], f["line"])
            f["memory_id"] = m["id"]
            f["fname"] = fname
            # Keep heuristic uncertain (from UNC regex). Do NOT use gold labels.
            # (N4.GOLD_F includes "uncertain" because N4 strips gold before clf predicts;
            #  here the UNC-derived flag IS the shadow substitute for that clf.)
            f["prereg"] = False
            str_facts.append(f)
    # Merge: prefer string extract entries; add numeric-only if missing
    seen = set()
    out = []
    for f in str_facts + numeric:
        key = (f.get("memory_id"), f.get("entity"), f.get("attr"), str(f.get("value")))
        if key in seen:
            continue
        seen.add(key)
        # Recompute uncertain from raw/text markers only (no gold)
        raw = str(f.get("raw") or "") + " " + str(f.get("value") or "")
        f["uncertain"] = bool(UNC.search(raw))
        f["prereg"] = False
        out.append(f)
    for i, f in enumerate(out):
        f["id"] = i
    return out


# ---------- NL question → structured q ----------
def _tok(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def map_question(question: str, facts: List[dict]) -> Optional[dict]:
    """Best-effort entity/attr mapping via token overlap. Honest, not fluent."""
    q = question.strip()
    ql = q.lower()
    qt = set(_tok(q))
    # strip question stopwords
    stop = {
        "what", "who", "when", "where", "which", "how", "did", "does", "do", "is", "are",
        "was", "were", "the", "a", "an", "of", "to", "in", "on", "for", "and", "or",
        "by", "with", "from", "its", "it", "be", "mean", "means", "define", "many",
        "year", "color", "does", "well",
    }
    qsig = qt - stop

    # Heuristic attr hints from question phrasing
    attr_hints = []
    if "capital" in ql:
        attr_hints.append("capital")
    if "wrote" in ql or "author" in ql:
        attr_hints.append("author")
    if "discover" in ql:
        attr_hints.append("discoverer")
    if "invent" in ql:
        attr_hints.append("inventor")
    if "found" in ql:
        attr_hints.append("founder")
    if "atomic" in ql:
        attr_hints.append("atomic number")
    if "moon" in ql:
        attr_hints.append("moons")
    if "color" in ql:
        attr_hints.append("color")
    if "confident-wrong" in ql or "confident wrong" in ql:
        attr_hints.append("confident-wrong")
    if "accuracy" in ql or "hybrid" in ql or "retrieval" in ql or "cad" in ql:
        attr_hints.append("accuracy")
    if "kappa" in ql or "lambda" in ql or "probe metric" in ql:
        attr_hints.append("value")
    if "tutor" in ql or "teacher" in ql:
        attr_hints.append("verdict")
    if "hypothalamus" in ql:
        attr_hints.append("supported")
    if "callosum" in ql or "degree shuffle" in ql or "transfer" in ql:
        attr_hints += ["primary result", "beat exact on confident-wrong"]
    if "stp" in ql or "ablation" in ql or "beat base" in ql:
        attr_hints.append("best arm vs base")
    if "neuron-5" in ql or "neuron 5" in ql:
        attr_hints.append("beat e-auto on confident-wrong")
    if "qms" in ql:
        attr_hints.append("hardware case")
    if "label" in ql or "human judgment" in ql:
        attr_hints.append("human judgment")

    best = None
    best_score = 0.0
    for f in facts:
        ent_t = set(_tok(f["entity"]))
        attr_t = set(_tok(f["attr"]))
        # overlap
        score = 2.0 * len(qsig & ent_t) + 1.5 * len(qsig & attr_t)
        # value tokens sometimes appear in "how many moons" style — small bonus
        score += 0.25 * len(qsig & set(_tok(str(f["value"]))))
        for h in attr_hints:
            if h == f["attr"] or h in f["attr"]:
                score += 3.0
            if set(_tok(h)) <= attr_t:
                score += 2.0
        # entity mention in question
        if f["entity"] in ql:
            score += 4.0
        # distinctive entity tokens (length>4) present in question — break ties like kappa vs lambda
        for tok in ent_t:
            if len(tok) > 4 and tok in qsig and tok not in {"shadow", "probe", "metric", "value"}:
                score += 5.0
        if score > best_score:
            best_score = score
            best = f
    if best is None or best_score < 2.0:
        return None
    return {
        "text": q,
        "intent": "value",
        "project": best["project"],
        "entity": best["entity"],
        "attr": best["attr"],
        "sattr": None,
        "sval": None,
        "_map_score": best_score,
        "_map_fact": best["id"],
    }


def hit_gold(ans: Optional[str], gold: str) -> Tuple[bool, bool, bool]:
    """Return (correct, abstain, wrong) vs gold string. Gold 'ABSTAIN' expects None."""
    if ans is None:
        abstain = True
        correct = gold.strip().upper() == "ABSTAIN"
        return correct, abstain, False if correct else False  # wrong=False on abstain
    # answered
    if gold.strip().upper() == "ABSTAIN":
        return False, False, True
    a = str(ans).strip().lower()
    g = str(gold).strip().lower()
    # numeric compare when both numeric
    na, ng = N.norm_num(a), N.norm_num(g)
    if na is not None and ng is not None:
        ok = na == ng
        return ok, False, not ok
    if g in a or a in g:
        return True, False, False
    # token overlap soft match for short answers
    gt, at = set(_tok(g)), set(_tok(a))
    if gt and gt <= at:
        return True, False, False
    if gt and at and len(gt & at) / len(gt) >= 0.6:
        return True, False, False
    return False, False, True


class ShadowEAuto:
    def __init__(self, views: List[dict], facts: List[dict], seed: int = 0):
        self.views = views
        self.facts = facts
        self.seed = seed
        self.limitation = LIMITATION
        edges = N.build_edges(facts)
        # Graph E mode — uncertain already set heuristically
        self.graph = N.Graph("E", views, facts, edges, "E", seed, cap=None)
        self.fmap = defaultdict(list)
        for f in facts:
            self.fmap[f["memory_id"]].append(f)
        self.n_numeric_native = sum(
            1 for f in facts if N.norm_num(str(f["value"])) is not None
        )
        self.n_string = len(facts) - self.n_numeric_native

    @classmethod
    def from_knowledge(
        cls,
        upload_path: Path = UPLOAD_JSONL,
        notes_path: Path = NOTES_JSONL,
        seed: int = 0,
        include_uploads: bool = True,
        include_notes: bool = True,
    ) -> "ShadowEAuto":
        rows: List[dict] = []
        if include_uploads:
            rows.extend(_load_jsonl(upload_path))
        if include_notes:
            rows.extend(_load_jsonl(notes_path))
        views = [render_memory(r, i) for i, r in enumerate(rows)]
        # drop helper keys before N4 sees them — auto_extract only needs id/text/path/project
        clean_views = [
            {"id": v["id"], "text": v["text"], "project": v["project"], "path": v["path"]}
            for v in views
        ]
        facts = build_facts(clean_views)
        return cls(clean_views, facts, seed=seed)

    def answer(self, question: str) -> dict:
        """Answer via E-auto or ABSTAIN. Never raises into primary path."""
        qstruct = map_question(question, self.facts)
        if qstruct is None:
            return {
                "answer": None,
                "display": "ABSTAIN",
                "confidence": 0.0,
                "reason": "no entity/attr map from NL question",
                "mapped": None,
            }
        # Install fmap for Graph (same as N4 ask())
        N.FACTS_BY_MEM.clear()
        N.FACTS_BY_MEM.update(self.fmap)
        self.graph.read = 0
        # Pass only non-gold fields (q_view spirit)
        q_view = {k: qstruct.get(k) for k in ("text", "intent", "project", "entity", "attr", "sattr", "sval")}
        ans, conf = self.graph.answer(q_view)
        reason = "answered"
        if ans is None:
            if conf <= 0.2:
                reason = "abstain: only uncertain/proposed support (E)"
            elif conf <= 0.3:
                reason = "abstain: unresolved contradiction or weak support (E)"
            else:
                reason = f"abstain: conf={conf:.3f}"
        return {
            "answer": ans,
            "display": "ABSTAIN" if ans is None else str(ans),
            "confidence": float(conf),
            "reason": reason,
            "mapped": {
                "entity": qstruct["entity"],
                "attr": qstruct["attr"],
                "project": qstruct["project"],
                "map_score": qstruct.get("_map_score"),
            },
        }

    def answer_sidecar(self, question: str) -> str:
        r = self.answer(question)
        mapped = r.get("mapped") or {}
        return (
            f"[shadow E-auto] {r['display']}  "
            f"(conf={r['confidence']:.3f}; {r['reason']}; "
            f"map={mapped.get('entity')!r}/{mapped.get('attr')!r})"
        )

    def info(self) -> dict:
        return {
            "n_memories": len(self.views),
            "n_facts": len(self.facts),
            "n_numeric_values": self.n_numeric_native,
            "n_string_values": self.n_string,
            "project": PROJECT,
            "limitation": LIMITATION,
            "neuron4_import": str(N4_ROOT / "run_neuron4.py"),
        }


def format_sidecar_line(result: dict) -> str:
    mapped = result.get("mapped") or {}
    return (
        f"[shadow E-auto] {result['display']}  "
        f"(conf={result['confidence']:.3f}; {result['reason']}; "
        f"map={mapped.get('entity')!r}/{mapped.get('attr')!r})"
    )
