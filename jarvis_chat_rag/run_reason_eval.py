#!/usr/bin/env python3
"""Cheap CPU-only evaluation. No model checkpoint load, training, or remote calls.

Gold fields are used ONLY here, never by the reasoner. Exact substring scoring
is inherited from the small historical benchmark; no support/trace credit.
"""
import hashlib
import json
import os
import time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')

from answer_modes import answer_one_mode
from retriever import ChatRetriever
from pack_router import RoutedRetriever
from extractive import best_extractive
from dictionary_lookup import answer_define, extract_define_query
from reason_sort import VERSION

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'


def correct(answer, gold, abstain=False):
    if gold.upper() == 'ABSTAIN':
        return abstain
    return not abstain and gold.casefold() in answer.casefold()


def metrics(rows, prefix='reason'):
    n = len(rows)
    answered = [r for r in rows if not r[prefix + '_abstain']]
    wrong = sum(not r[prefix + '_correct'] for r in answered)
    return {'n': n, 'correct': sum(r[prefix + '_correct'] for r in rows),
            'accuracy': sum(r[prefix + '_correct'] for r in rows) / max(1, n),
            'answered': len(answered), 'coverage': len(answered) / max(1, n),
            'confident_wrong_among_answered': wrong / len(answered) if answered else None,
            'wrong_answered': wrong}


def evaluate(items, retriever):
    rows = []
    for item in items:
        q, gold = item['question'], str(item['gold'])
        result = answer_one_mode(None, None, retriever, q, mode='reason')
        hits = retriever.search(q, k=retriever.k)
        ext = best_extractive(q, hits)
        definition = answer_define(q) if extract_define_query(q) else None
        baseline = definition or (ext or {}).get('span') or ''
        ba = not bool(baseline)
        ret = None if gold.upper() == 'ABSTAIN' else gold.casefold() in '\n'.join(h['text'] for h in hits).casefold()
        rows.append({**item, 'reason_answer': result['answer'],
                     'reason_abstain': result['abstain'], 'decision': result['decision'],
                     'reason_correct': correct(result['answer'], gold, result['abstain']),
                     'trace': result['trace'], 'support': result.get('support'),
                     'groups': result['groups'], 'baseline_answer': baseline,
                     'baseline_abstain': ba, 'baseline_correct': correct(baseline, gold, ba),
                     'retrieval_hit': ret, 'retrieved_ids': [h.get('id') for h in hits]})
    return {'summary': metrics(rows), 'live_extractive_wordnet_baseline': metrics(rows, 'baseline'),
            'live_retrieval': {'n': sum(r['retrieval_hit'] is not None for r in rows),
                               'hit_rate': sum(bool(r['retrieval_hit']) for r in rows) / max(1, sum(r['retrieval_hit'] is not None for r in rows))},
            'items': rows}


def main():
    start = time.monotonic()
    full = json.loads((OUT / 'full_eval.json').read_text())
    shadow = json.loads((OUT / 'shadow_eval.json').read_text())
    # Reuse exact question/gold sets from the prior reports; never pass metadata
    # to reason_answer. Existing index already strips eval fields from hits.
    keys = ('id', 'kind', 'question', 'gold')
    items = [{k: r.get(k) for k in keys} for r in full['items']]
    notes = [{k: r.get(k) for k in keys} for r in shadow['jarvis_notes']['items']]
    retriever = RoutedRetriever(ChatRetriever(k=3, max_chars=900))
    results = {'version': VERSION,
               'finished_et': datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S ET'),
               'default': 'cad_hybrid (unchanged; reason is opt-in)',
               'full_eval_30q': evaluate(items, retriever),
               'jarvis_notes': evaluate(notes, retriever),
               'published_baselines_not_rerun': {
                   'full_eval_30q': full['summary'],
                   'eauto_30q': shadow['full_eval_30q']['summary'],
                   'eauto_jarvis_notes': shadow['jarvis_notes']['summary']},
               'scoring': 'Gold substring in answer ONLY (not support or trace). ABSTAIN requires explicit abstention. Any nonabstain wrong answer counts as confident-wrong; not calibrated confidence.',
               'caveats': [
                   'Small existing development sets, not independent validation; no significance claim. Iterative regression fixes were made after inspecting these sets; intermediate runs retained as reason_eval_development_v0/v1/v2.json.',
                   'Default stays CAD hybrid despite meeting the published 30Q number: notes coverage/routing still fails, and no independent holdout validates the rule changes.',
                   'Historical .900 hybrid matching included supporting sentences (and global mode generation); current reason score is stricter answer-only. Historical numbers are context, not a controlled head-to-head.',
                   'WordNet remains enabled; historical hybrid .900 did not use all automatic WordNet answers. Fresh extractive+WordNet baseline is shown to avoid attributing dictionary improvements to reasoning.',
                   'Prior shadow_eval jarvis_notes hybrid_acc is an optimistic retrieval-or-span surrogate; do not call it answer accuracy. This run reports actual span-only baseline.',
                   'Same synthetic notes supplied question/golds; no independent source validation.',
                   'Only supplied top-3 hits checked. Conflicts outside retrieval, paraphrased slots, negation scope, multi-hop reasoning, and numeric comparisons may be missed.',
                   'NOT SUPPORTED is a reported negative verdict, not automatically an uncertain number; exploratory/PARTIAL statements retain caveats in supporting evidence.',
                   'This sorts/checks retrieved text, not general intelligence, consciousness, or learning.'],
               'safety': {'no_qpu': True, 'no_training': True, 'no_model_load': True,
                          'no_kaggle_upload': True, 'no_institutional_books_indexed': True}}
    root = Path('/workspace/jarvis/neuron/neuron4')
    before_path = OUT / 'reason_neuron4_before.json'
    before = json.loads(before_path.read_text())
    after = {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
             for f in root.rglob('*') if f.is_file()}
    results['safety']['neuron4_all_files_unchanged'] = before == after
    results['safety']['neuron4_files_hashed'] = len(after)
    results['seconds'] = round(time.monotonic() - start, 3)
    results['artifact_sha256'] = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in ('reason_sort.py', 'extractive.py', 'answer_modes.py', 'chat.py', 'run_reason_eval.py', 'test_reason_sort.py', 'results/full_eval.json', 'results/shadow_eval.json', 'knowledge/jarvis_notes.jsonl')}
    import io
    import unittest
    test_log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName('test_reason_sort')
    tested = unittest.TextTestRunner(stream=test_log).run(suite)
    results['tests'] = {'regression_command': 'python3 -m unittest -v test_reason_sort',
                        'run': tested.testsRun, 'failures': len(tested.failures),
                        'errors': len(tested.errors), 'successful': tested.wasSuccessful(),
                        'log': test_log.getvalue()}
    (OUT / 'reason_eval.json').write_text(json.dumps(results, indent=2) + '\n')
    lines = ['# Training-free evidence reason/sort evaluation', '',
             f"Finished: {results['finished_et']} (America/New_York)", '',
             '**Default unchanged: CAD hybrid.** Use `--reason` or `--mode reason`; REPL `/reason`.', '',
             'No checkpoint loaded. No generation rescored. No gold metadata enters the reasoner.', '',
             '| Set / method | n | Accuracy | Coverage | Wrong among answered |',
             '|---|---:|---:|---:|---:|']
    for setname in ('full_eval_30q', 'jarvis_notes'):
        for key, label in [('summary', 'reason'), ('live_extractive_wordnet_baseline', 'live extractive + WordNet')]:
            s = results[setname][key]
            lines.append(f"| {setname} / {label} | {s['n']} | {s['accuracy']:.3f} | {s['coverage']:.3f} | {s['confident_wrong_among_answered']:.3f} |")
        se = shadow[setname]['summary']
        lines.append(f"| {setname} / published E-auto shadow | {se['n']} | {se['eauto_accuracy']:.3f} | {se['eauto_coverage']:.3f} | {se['eauto_confident_wrong_among_answered']:.3f} |")
        ret = results[setname]['live_retrieval']
        lines.append(f"| {setname} / live retrieval (gold anywhere, not answer accuracy) | {ret['n']} | {ret['hit_rate']:.3f} | n/a | n/a |")
    lines += ['', 'Historical jarvis_notes retrieval: .923 on 13 answerable questions (global, not current routed retrieval). The prior .923 hybrid figure used retrieval-or-span matching and is not actual answer accuracy.', '', 'Historical 30Q: hybrid **.900**, routed retrieval **.967**, gen **.300**, CAD **.433**. These are published, not rerun.', '',
              '## Scoring', '', results['scoring'], '', '## Per-question outputs (answer only)', '',
              '| Set | ID | Gold | Answer | Decision | Correct |', '|---|---|---|---|---|---|']
    for name in ('full_eval_30q', 'jarvis_notes'):
        for r in results[name]['items']:
            ans = (r['reason_answer'] or '[abstain]').replace('\n', ' / ').replace('|', '/')
            lines.append(f"| {name} | {r['id']} | {r['gold']} | {ans} | {r['decision']} | {r['reason_correct']} |")
    lines += ['', '## Example evidence traces', '']
    for r in results['full_eval_30q']['items'] + results['jarvis_notes']['items']:
        if r['id'] in {'paine', 'jn_probe_kappa_a', 'jn_probe_lambda'}:
            lines += [f"### {r['question']}", f"Answer: {r['reason_answer'] or '[abstain]'}", ''] + ['- ' + t for t in r['trace']] + ['']
    lines += ['## Honest limitations / comparison caveats', ''] + ['- ' + c for c in results['caveats']]
    lines += ['', '## Tests', '', results['tests']['log'], '', '## Safety', '', json.dumps(results['safety'], indent=2), '',
              '## Run', '', '```bash', 'cd /workspace/jarvis/chat_rag && nice -n 19 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 chat.py --reason --once "What is shadow probe metric Kappa?"', '```', '']
    (OUT / 'reason_eval.md').write_text('\n'.join(lines))
    for name in ('full_eval_30q', 'jarvis_notes'):
        print(name, json.dumps(results[name]['summary']))
    print('Safety:', results['safety'])


if __name__ == '__main__':
    main()
