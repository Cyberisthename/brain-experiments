"""Mechanical unit checks; fixtures are numbers, not scientific claims."""
import unittest
from collections import Counter
from system import extract,edges_for,degree,shuffled,Reasoner,norm
class Tests(unittest.TestCase):
    def test_extraction_offsets_and_decimals(self):
        cs=[dict(id='a',text='Test first metric was 0.125. Test second metric was 0.75.')]
        fs=extract(cs)
        self.assertEqual([f['value'] for f in fs],['0.125','0.75'])
        for f in fs:self.assertEqual(cs[0]['text'][f['span'][0]:f['span'][1]],f['raw'])
    def test_degree_and_label_invariants(self):
        es=[(0,1,'same_card'),(1,2,'contradicts'),(2,3,'unrelated_similar'),(3,0,'supported')]
        for seed in range(8):
            b=shuffled(es,'degree_shuffle',seed);c=shuffled(es,'label_shuffle',seed)
            self.assertEqual(degree(b),degree(es));self.assertEqual(len(b),len(es))
            self.assertEqual(Counter(x[2] for x in c),Counter(x[2] for x in es))
            self.assertEqual([x[:2] for x in c],[x[:2] for x in es])
    def test_no_graph_bypass_and_local_bag(self):
        cs=[dict(id='a',text='Test first metric was 0.125. Test second metric was 0.75.')]
        fs=extract(cs);q=dict(text=cs[0]['text'],project='uploaded',slot=fs[1]['slot'],seed_slot=fs[0]['slot'],seed_value=fs[0]['value'])
        g=Reasoner(cs,fs,'E');self.assertEqual(g.answer(q)['answer'],'0.75')
        g.adj.clear();self.assertIsNone(g.answer(q)['answer'])
        self.assertEqual(Reasoner(cs,fs,'bag').answer(q)['answer'],'0.75')
    def test_boundary(self):
        cs=[dict(id='a',text='Test weak metric is ~0.55 (approx, unverified).')];fs=extract(cs)
        self.assertTrue(fs[0]['weak'])
        q=dict(text=cs[0]['text'],project='uploaded',slot=fs[0]['slot'])
        self.assertIsNone(Reasoner(cs,fs,'E').answer(q)['answer'])
        self.assertEqual(Reasoner(cs,fs,'bag').answer(q)['answer'],'0.55')
    def test_gold_rejected(self):
        r=Reasoner([],[],'bag')
        with self.assertRaises(AssertionError):r.answer(dict(text='x',gold='x'))
if __name__=='__main__':unittest.main()
