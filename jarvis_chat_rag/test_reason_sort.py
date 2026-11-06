"""Small deterministic regression/probe tests; not a held-out intelligence test."""
import unittest
from unittest.mock import patch
from reason_sort import reason_answer
from answer_modes import answer_one_mode, normalize_mode


def hit(text, score=20, **metadata):
    return {'text': text, 'source': 'upload', 'score': score, **metadata}


class ReasonSortTests(unittest.TestCase):
    def ask(self, q, hits):
        return reason_answer(q, hits, prefer_dict=False)

    def test_conflicting_scalar(self):
        r = self.ask('What is sensor Omega output?', [hit('Sensor Omega output is 0.12.'), hit('Sensor Omega output is 0.87.', 2)])
        self.assertEqual(r['decision'], 'conflict')
        self.assertTrue(r['abstain'])

    def test_uncertain_value(self):
        for marker in ('~', 'approximately ', 'estimated '):
            r = self.ask('What is sensor Sigma output?', [hit(f'Sensor Sigma output is {marker}0.62.')])
            self.assertEqual(r['decision'], 'uncertain')

    def test_sort_before_bm25(self):
        r = self.ask('Who wrote Silent Harbor?', [hit('Anne Vale wrote Silent Winter.', 99), hit('Jon Reed wrote Silent Harbor.', 20)])
        self.assertEqual(r['answer'], 'Jon Reed')

    def test_repeated_conflict_not_majority(self):
        r = self.ask('What is the capital of Velora?', [hit('The capital of Velora is Arven. ' * 8), hit('The capital of Velora is Bekar.', 2)])
        self.assertEqual(r['decision'], 'conflict')

    def test_numbered_entity(self):
        r = self.ask('What was SYSTEM-2 accuracy?', [hit('SYSTEM-3 accuracy was 0.99.', 99), hit('SYSTEM-2 accuracy was 0.61.', 20)])
        self.assertEqual(r['answer'], '0.61')

    def test_no_evidence(self):
        self.assertTrue(self.ask('Who wrote Lost Harbor?', [])['abstain'])
        self.assertTrue(self.ask('What is the capital of Velora?', [hit('Sand dissolves slowly in water.')])['abstain'])

    def test_no_gold_metadata_leakage(self):
        text = 'The capital of Velora is Arven.'
        base = self.ask('What is the capital of Velora?', [hit(text)])
        poisoned = self.ask('What is the capital of Velora?', [hit(text, answer='Bekar', question='OTHER', fact_statement='The capital is Bekar.', gold='Bekar')])
        self.assertEqual(base['answer'], poisoned['answer'])
        self.assertEqual(base['decision'], poisoned['decision'])

    def test_negative_verdict_not_uncertain_number(self):
        r = self.ask('How did the absorber test go?', [hit('The absorber test was NOT SUPPORTED.')])
        self.assertFalse(r['abstain'])
        self.assertEqual(r['answer'], 'NOT SUPPORTED')

    def test_passive_relation(self):
        for verb in ('invented', 'founded', 'discovered'):
            r = self.ask(f'Who {verb} the Heliad engine?', [hit(f'The Heliad engine was {verb} by Ari Hale.')])
            self.assertEqual(r['answer'], 'Ari Hale')

    def test_author_name_also_in_title(self):
        r = self.ask('Who wrote The Chronicle of Tamara?', [hit('Tamara wrote The Chronicle of Tamara.')])
        self.assertEqual(r['answer'], 'Tamara')

    def test_metric_modifier(self):
        r = self.ask('What was adapter accuracy on the eval?', [hit('The control eval accuracy was 0.9.', 99), hit('The adapter accuracy was 0.7.', 20)])
        self.assertEqual(r['answer'], '0.7')

    def test_partial_value_overlap_is_conflict(self):
        r = self.ask('What is the capital of Velora?', [hit('The capital of Velora is Arven.'), hit('The capital of Velora is Arven or Bekar.')])
        self.assertEqual(r['decision'], 'conflict')

    def test_count_is_not_entity_version(self):
        r = self.ask('How many moons does Satellite-2 have?', [hit('Planet Satellite-2 has seven moons.')])
        self.assertIn('seven', r['answer'])

    def test_run_outcome_not_definition(self):
        r = self.ask('How did the teacher training go?', [hit('Knowledge distillation trains students to reproduce a teacher model.')])
        self.assertTrue(r['abstain'])

    def test_dictionary_preserved(self):
        with patch('reason_sort.answer_define', return_value='A test definition'):
            r = reason_answer('Define neuron', [])
        self.assertEqual(r['decision'], 'dictionary')

    def test_reason_mode_never_generates(self):
        class Retriever:
            k = 3
            def search(self, query, k):
                return [hit('The capital of Velora is Arven.')]
        with patch('answer_modes.generate_cad', side_effect=AssertionError('model called')):
            r = answer_one_mode(None, None, Retriever(), 'What is the capital of Velora?', mode='reason')
        self.assertEqual(r['answer'], 'Arven')
        self.assertEqual(normalize_mode('default'), 'cad_hybrid')


if __name__ == '__main__':
    unittest.main()
