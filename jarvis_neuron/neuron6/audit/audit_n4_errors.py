"""Exploratory audit of frozen NEURON-4 E_auto per-question outputs. Read-only. Not a NEURON-6 test.
Reason codes come from E's read rule (n3lib Graph.read_facts, mode E) via its returned confidence:
0.0 = no usable candidate (not retrieved / not in project / hop bridge failed), 0.2 = only uncertain-predicted support,
0.3 = unresolved contradiction among certain facts, >=0.6 = answered."""
import json, collections, hashlib
SRC = "/workspace/jarvis/neuron/neuron4/results/per_question.jsonl"
raw = open(SRC, "rb").read(); P = [json.loads(l) for l in raw.decode().splitlines()]
def reason(r):
    s = r["E_auto"]
    if s["answer"] != "ABSTAIN": return "answered"
    return {0.0: "abstain_no_candidate", 0.2: "abstain_only_uncertain", 0.3: "abstain_contradiction"}.get(round(s["confidence"], 1), "abstain_other")
out = {"source_sha256": hashlib.sha256(raw).hexdigest()}
for c in (1, 2):
    R = [r for r in P if r["corpus"] == c]; n = len(R)
    ans = [r for r in R if r["E_auto"]["answer"] != "ABSTAIN"]; wrong = [r for r in ans if r["E_auto"]["wrong"]]
    ansable = [r for r in R if r["gold"] != "ABSTAIN"]; absq = [r for r in R if r["gold"] == "ABSTAIN"]
    d = dict(n_questions=n, n_answered=len(ans), n_confident_wrong=len(wrong),
             confident_wrong_over_all=len(wrong)/n, wrong_given_answered=len(wrong)/max(1, len(ans)),
             n_answerable=len(ansable), answerable_answered=sum(r["E_auto"]["answer"] != "ABSTAIN" for r in ansable),
             answerable_answered_wrong_value=sum(r["E_auto"]["wrong"] for r in ansable),
             n_gold_abstain=len(absq), gold_abstain_but_answered=sum(r["E_auto"]["answer"] != "ABSTAIN" for r in absq))
    # error taxonomy (only categories the logs support)
    tax = collections.Counter(); ex = collections.defaultdict(list)
    for r in R:
        rs = reason(r); k = r["kind"]; g = r["gold"]
        if g == "ABSTAIN" and rs == "answered":
            cat = {"uncertain": "A_missed_weak_evidence (answered; gold=uncertain-cue fact; uncertainty classifier miss)",
                   "conflict": "B_missed_conflict (answered; gold=conflicting certain values)"}[k]
        elif g != "ABSTAIN" and rs == "answered" and r["E_auto"]["wrong"]:
            cat = f"C_wrong_value_on_answerable ({k})"
        elif g != "ABSTAIN" and rs != "answered":
            cat = {"abstain_contradiction": "D_false_conflict (abstained as contradiction; gold has single answer)",
                   "abstain_only_uncertain": "E_false_weak_evidence (classifier flagged; gold certain)",
                   "abstain_no_candidate": "F_retrieval_or_scope_miss (no usable candidate / hop bridge failed)"}.get(rs, "G_other_abstain") + f" [{k}]"
        else: continue
        tax[cat] += 1
        if len(ex[cat]) < 3: ex[cat].append(dict(seed=r["seed"], qid=r["qid"], q=r["question"][:160], gold=g, answer=r["E_auto"]["answer"], conf=r["E_auto"]["confidence"]))
    d["error_taxonomy_counts"] = dict(tax.most_common()); d["examples"] = ex
    d["by_kind"] = {k: dict(n=len(x := [r for r in R if r["kind"] == k]), answered=sum(r["E_auto"]["answer"] != "ABSTAIN" for r in x),
                     wrong=sum(r["E_auto"]["wrong"] for r in x), correct=sum(r["E_auto"]["correct"] for r in x),
                     abstain_reasons=dict(collections.Counter(reason(r) for r in x)))
                    for k in sorted({r["kind"] for r in R})}
    out[f"corpus{c}"] = d
json.dump(out, open("n4_error_audit.json", "w"), indent=1)
for c in (1, 2):
    d = out[f"corpus{c}"]; print(f"\n== corpus {c}: {d['n_confident_wrong']}/{d['n_questions']} conf-wrong over all = {d['confident_wrong_over_all']:.3f}; {d['n_confident_wrong']}/{d['n_answered']} wrong given answered = {d['wrong_given_answered']:.3f}")
    print(f"   answerable {d['n_answerable']}: answered {d['answerable_answered']}, wrong value {d['answerable_answered_wrong_value']}; gold-abstain {d['n_gold_abstain']}: answered anyway {d['gold_abstain_but_answered']}")
    for k, v in d["error_taxonomy_counts"].items(): print(f"   {v:4d}  {k}")
