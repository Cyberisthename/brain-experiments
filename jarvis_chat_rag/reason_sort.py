"""Deterministic evidence sorter/checker, not LM reasoning or a truth oracle.

Only question + hit text/source/score are used. Evaluation metadata (answer,
question, fact_statement, ids) cannot influence decisions. IDs are trace labels.
Conflicts can only be found inside the supplied retrieval window.
"""
from __future__ import annotations

import re
from extractive import best_extractive, _query_terms
from dictionary_lookup import answer_define, extract_define_query

VERSION = 'reason-sort-v1'
# Small lexical normalization, not learned and not benchmark entity mappings.
ALIASES = {'generation': 'gen', 'routed': 'route', 'routing': 'route',
           'retrieval': 'retrieve', 'judgments': 'judgment', 'judgements': 'judgment',
           'labels': 'label', 'questions': 'question', 'tests': 'test',
           'training': 'train', 'explanations': 'explanation', 'began': 'begin',
           'started': 'begin', 'written': 'wrote', 'authored': 'wrote',
           'teacher': 'tutor'}
EXTRA_STOP = {'s', 'go', 'did', 'all', 'four', 'up', 'hold', 'than', 'worth', 'yet', 'have', 'has', 'many'}
UNCERTAIN = re.compile(r'~\s*\d|\b(?:approx(?:imate(?:ly)?)?|est\.?|estimated|uncertain|unverified|unknown|perhaps|possibly|might)\b', re.I)
QUALIFIED = re.compile(r'\b(?:partial|exploratory|not supported|falsified)\b', re.I)


def terms(text):
    # Legacy tokenizer retains sentence-final dots, which otherwise makes
    # Harbor? and Harbor. look like different entities to the sorter.
    tokens = [t.strip(".,;:!?\"'()[]") for t in _query_terms(text)]
    from extractive import STOP
    return {ALIASES.get(t, t) for t in tokens if t and t not in EXTRA_STOP and t not in STOP}


def sentences(text):
    # Don't split Dr./Mr. names. The span scorer remains the existing scorer.
    safe = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof)\.', r'\1<dot>', text)
    return [s.replace('<dot>', '.').strip() for s in
            re.split(r'(?<=[.!?])\s+|\n+', safe) if len(s.strip()) >= 8]


def slot_value(text):
    """Conservative relation keys: same subject+attribute, not any two numbers.

    No numeric comparison across sample sizes, different arms, or different
    entity versions. Unparsed relations are *not* declared conflict-free truth.
    """
    text = text.strip().rstrip('.')
    m = re.match(r'(.+?)\s+(?:wrote|authored)\s+(.+)$', text, re.I)
    if m:
        return ('author', tuple(sorted(terms(m[2])))), m[1].strip()
    m = re.match(r'(.+?)\s+(?:was written|was authored)\s+by\s+(.+)$', text, re.I)
    if m:
        return ('author', tuple(sorted(terms(m[1])))), m[2].strip()
    for verb in ('discovered', 'invented', 'founded'):
        # Passive must precede active: 'was invented by' contains 'invented'.
        m = re.match(r'(.+?)\s+was\s+' + verb + r'\s+by\s+(.+)$', text, re.I)
        if m:
            return (verb, tuple(sorted(terms(m[1])))), m[2].strip()
        m = re.match(r'(.+?)\s+' + verb + r'\s+(.+)$', text, re.I)
        if m:
            return (verb, tuple(sorted(terms(m[2])))), m[1].strip()
    m = re.match(r'(.+?)\s+(did not beat|beat)\s+(.+)$', text, re.I)
    if m:
        return ('beat', tuple(sorted(terms(m[1]))), tuple(sorted(terms(m[3])))), ('no' if m[2].lower().startswith('did') else 'yes')
    m = re.match(r'(.+?)\s+(?:is|are|was|were)\s+(.+)$', text, re.I)
    if m:
        subject, value = m.groups()
        # For explicit scalar assignments ignore trailing population/context.
        n = re.match(r'(~?\s*[+-]?\d+(?:\.\d+)?)\b', value)
        if n:
            value = n[1].replace(' ', '')
        else:
            value = re.split(r'[;(]', value)[0].strip()
        return ('copula', tuple(sorted(terms(subject)))), value
    return None, None


def value_key(value):
    ts = terms(value)
    return tuple(sorted(ts)) or value.lower().strip(' .')


def equivalent(a, b, allow_surname=False):
    # Full name vs surname is not a conflict (Shakespeare / William Shakespeare).
    aa, bb = terms(a), terms(b)
    return value_key(a) == value_key(b) or (allow_surname and bool(aa and bb) and (aa <= bb or bb <= aa))


def pack(source):
    if source == 'jarvis_notes':
        return 'jarvis_notes'
    if source in {'novel', 'upload', 'fact_card', 'pack_fact'}:
        return 'uploads'
    if source == 'wordnet':
        return 'dictionary'
    return 'books'


def polarity(question, sentence):
    """Surface yes/no only when an aligned statement explicitly says so."""
    if not re.match(r'^(did|does|do|is|are|was|were|has|have|can)\b', question, re.I):
        return None
    if re.search(r'\b(?:not|no|never|neither|falsified|lost)\b', sentence, re.I):
        return 'no'
    if re.search(r'\b(?:beat|supported|true)\b', sentence, re.I):
        return 'yes'
    return None


def reason_answer(question, hits, min_ext=1.5, prefer_dict=True):
    """Return answer, decision, extractive, fact groups, and short evidence trace.

    Trace is a reproducible evidence audit, not private LM chain-of-thought.
    """
    if prefer_dict and extract_define_query(question):
        definition = answer_define(question)
        if definition:
            return _result(question, definition, 'dictionary', None,
                           ['Used WordNet definition; passage reasoning bypassed.'], [])
        return _result(question, '', 'weak', None,
                       ['Skipped dictionary miss; no fallback guess.'], [])
    qt = terms(question)
    max_bm = max([float(h.get('score') or 0) for h in hits] + [1.0])
    facts, seen = [], set()
    for h in hits:
        for sent in sentences(h.get('text') or ''):
            # Dedup repeated text per source, but keep independent conflicting text.
            key = (h.get('source'), sent.lower())
            if key in seen:
                continue
            seen.add(key)
            st = terms(sent)
            cov = len(qt & st) / max(1, len(qt))
            # Numbered entity versions must not bleed into each other.
            q_ids = set(re.findall(r'\b[a-z]+[-_]\d+\b', question.lower()))
            s_ids = set(re.findall(r'\b[a-z]+[-_]\d+\b', sent.lower()))
            wrong_entity = bool(q_ids and s_ids and not q_ids <= s_ids)
            p = pack(h.get('source'))
            specificity = 1 / (1 + len(sent) / 180)
            rank = (4 * cov * cov + .25 * float(h.get('score') or 0) / max_bm
                    + .3 * specificity + (.08 if p in {'uploads', 'jarvis_notes'} else 0))
            slot, value = slot_value(sent)
            # Attribute modifier is a slot constraint, not background context:
            # 'adapter accuracy' must not silently become 'control accuracy'.
            modifier = re.search(r'\b([\w-]+)\s+(?:accuracy|rate|number)\b', question, re.I)
            wrong_slot = bool(slot and slot[0] == 'copula' and modifier and
                              terms(modifier[1]) and not terms(modifier[1]) <= set(slot[1]))
            if re.match(r'^how\b.*\bgo\??$', question.strip(), re.I):
                # A description of training is not a result of a training run.
                outcome = re.search(r'\b(?:supported|falsified|partial|result|pilot|score|lost|mean|finished|failed|passed|succeeded|accuracy|loss)\b', sent, re.I)
                wrong_slot = wrong_slot or not bool(outcome)
            facts.append({'sentence': sent, 'source': h.get('source'), 'id': h.get('id'),
                          'pack': p, 'coverage': cov, 'rank': rank,
                          'score': float(h.get('score') or 0), 'wrong_entity': wrong_entity, 'wrong_slot': wrong_slot,
                          'uncertain': bool(UNCERTAIN.search(sent)),
                          'qualified': bool(QUALIFIED.search(sent)),
                          'slot': slot, 'value': value})
    facts.sort(key=lambda f: f['rank'], reverse=True)
    eligible = [f for f in facts if f['coverage'] >= .55 and not f['wrong_entity'] and not f['wrong_slot']]
    trace = []
    for f in facts:
        if f not in eligible:
            why = ('different numbered entity' if f['wrong_entity'] else
                   'different requested attribute modifier' if f['wrong_slot'] else 'weak/off-topic query overlap')
            trace.append(f"Skipped {f['id']}: {why} ({f['coverage']:.0%}); {f['sentence'][:80]}")
            if len(trace) >= 2:
                break
    if not eligible:
        return _result(question, '', 'weak', None,
                       ['No sufficiently relevant statement in retrieved hits.'] + trace, [])
    top = eligible[0]
    # Related = same parsed slot, or close lexical support. Equal-slot members
    # are retained even at lower BM25 scores; repetition never resolves conflict.
    cluster = [f for f in eligible if (top['slot'] and f['slot'] == top['slot']) or
               (f['coverage'] >= top['coverage'] - .05 and
                len(terms(f['sentence']) & terms(top['sentence'])) /
                max(1, min(len(terms(f['sentence'])), len(terms(top['sentence'])))) >= .65)]
    groups = []
    for f in eligible:
        group_key = f['slot'] or ('overlap', tuple(sorted(qt & terms(f['sentence']))))
        group = next((g for g in groups if g['slot'] == group_key), None)
        if group is None:
            group = {'slot': group_key, 'facts': []}
            groups.append(group)
        group['facts'].append({k: f[k] for k in ['id', 'source', 'sentence', 'value', 'coverage', 'uncertain']})
    if top['slot']:
        values = [f for f in cluster if f['slot'] == top['slot'] and f['value']]
        if any(not equivalent(top['value'], f['value'], top['slot'][0] == 'author') for f in values):
            detail = '; '.join(f"{f['value']} ({f['id']})" for f in values)
            return _result(question, '', 'conflict', None,
                           [f'Values disagree for the same subject/attribute: {detail}.',
                            'Abstained; source rank/repetition cannot resolve disagreement.'] + trace, groups)
    if top['uncertain']:
        return _result(question, '', 'uncertain', None,
                       [f"Only uncertain top support: {top['id']}: {top['sentence']}",
                        'Abstained rather than turn an estimate into a verified value.'] + trace, groups)
    # Use the existing scorer on the highest-ranked consistent sentence rather
    # than all the long passages. Narrow scalar tails to avoid sample-size bleed.
    text = top['sentence']
    if re.search(r'\b(?:rate|accuracy|number|metric)\b', question, re.I):
        m = re.search(r'\b(?:is|was)\s+[+-]?\d+(?:\.\d+)?\b', text)
        if m:
            text = text[:m.end()] + '.'
    chosen_hit = {'text': text, 'source': top['source'], 'id': top['id'],
                  'score': top['score']}
    def allowed_span(span):
        st = terms(span)
        if not st:
            return False  # reject 'The' or 'In'
        slot, value = top['slot'], top['value']
        if slot and slot[0] == 'copula' and re.fullmatch(r'[+-]?\d+(?:\.\d+)?', value or ''):
            return span.strip() == value  # target value, never sample count/entity ID
        if question.lower().startswith('who ') and slot and slot[0] in {'author', 'discovered', 'invented', 'founded'}:
            name = terms(value) - {'dr', 'mr', 'mrs', 'ms', 'prof'}
            return bool(st) and st <= name
        if st <= qt:
            return False
        if re.search(r'\b(?:what year|which year|how many|atomic number)\b', question, re.I):
            return bool(re.fullmatch(r'[+-]?\d+(?:\.\d+)?', span.strip()) or st & {'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'})
        return True

    ext = best_extractive(question, [chosen_hit], min_score=min_ext, span_filter=allowed_span)
    if not ext:
        return _result(question, '', 'weak', None,
                       ['Relevant statement found, but no extractive span above threshold.'] + trace, groups)
    answer = ext['span']
    # Restore an honorific dropped by the legacy scorer's sentence splitter.
    match = re.search(r'\b(Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.)\s+' + re.escape(answer), top['sentence'])
    if match:
        answer = match[0]
    explicit_polarity = polarity(question, top['sentence'])
    if explicit_polarity:
        answer = explicit_polarity
    # Report explicit study verdicts as verdicts, not uncertain numeric claims.
    if re.match(r'^how\b', question, re.I):
        verdict = re.search(r'\b(?:NOT SUPPORTED|FALSIFIED|PARTIAL|SUPPORTED)\b', top['sentence'])
        if verdict:
            answer = verdict[0]
    trace.insert(0, f"Used {top['id']} [{top['pack']}, query overlap {top['coverage']:.0%}]: {top['sentence']}")
    trace.insert(1, f'Grouped {len(cluster)} related statement(s); no parsed same-slot conflict found.')
    if top['qualified']:
        trace.insert(2, 'Qualification retained: this is a reported study outcome, not proof of a general claim.')
    if len(eligible) > len(cluster):
        trace.append('Skipped other relevant statements: different slot or lower relevance.')
    result = _result(question, answer, 'answer', ext, trace, groups)
    result['support'] = top['sentence']
    # Supporting text is visible but is NOT used as the scored answer.
    result['display'] = f"[REASON] {answer}\n[SUPPORT] {top['sentence']}\n[EVIDENCE TRACE]\n" + '\n'.join('- ' + t for t in trace[:5])
    return result


def _result(question, answer, decision, ext, trace, groups):
    abstain = decision in {'weak', 'uncertain', 'conflict'}
    shown = answer if not abstain else {
        'weak': "I'm not sure: only weak or off-topic evidence.",
        'uncertain': "I'm not sure: the relevant value is uncertain/unverified.",
        'conflict': "I'm not sure: the retrieved values disagree."
    }[decision]
    return {'mode': 'reason', 'question': question, 'answer': answer,
            'abstain': abstain, 'decision': decision, 'extractive': ext,
            'generation': '', 'trace': trace[:5], 'groups': groups,
            'display': f'[REASON] {shown}\n[EVIDENCE TRACE]\n' + '\n'.join('- ' + t for t in trace[:5])}
