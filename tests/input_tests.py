import os
from nose.tools import eq_, nottest

from rungsted.feat_map import DictFeatMap
from rungsted.input import read_vw_seq

def vw_filename(fname):
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    return os.path.join(data_dir, fname)



def test_weighted_features():
    feat_map = DictFeatMap()
    seqs, labels = read_vw_seq(vw_filename('weighted.vw'), feat_map)

    expected = {b'1^a': 3, b'1^b': -3, b'1^c': 2.5, b'1^d': 1, b'1^e': 1E6}

    # Map feature ids to names
    index2feat = {idx: feat_name for feat_name, idx in feat_map.feat2index_.items()}
    lookup = {index2feat[index]: val for index, val in seqs[0].features}

    for key, val in expected.items():
        assert key in lookup
        eq_(lookup[key], val)

def test_cs_weights():
    seqs, labels = read_vw_seq(vw_filename('cs.vw'), DictFeatMap())
    eq_(len(seqs), 1)
    eq_(set(labels), set([b'A', b'B', b'C']))

    label_costs = seqs[0].label_costs

    # First token
    token_a = label_costs[0]
    eq_(labels[token_a[0][0]], b'C')
    eq_(token_a[0][1], 0.5)

    # Second token
    token_b = label_costs[1]
    eq_(labels[token_b[0][0]], b'A')
    eq_(token_b[0][1], 0.1)

    eq_(labels[token_b[1][0]], b'B')
    eq_(token_b[1][1], 0.2)

def test_importance_weights():
    seqs, labels = read_vw_seq(vw_filename('importance.vw'), DictFeatMap())

    eq_(len(seqs), 1, "One example")
    eq_(len(seqs[0]), 2, "Two tokens")

    imp1, imp2 = seqs[0].importance_weights
    eq_(imp1, 2.0)
    eq_(imp2, 1.0)

    # tok1 = seqs[0][0]
    # tok2 = seqs[0][1]

    # eq_(len(seqs), 1)
    # eq_(set(labels), set(['A', 'B', 'C']))
    #
    # label_costs = seqs[0].label_costs
    #
    # # First token
    # token_a = label_costs[0]
    # eq_(labels[token_a[0][0]], 'C')
    # eq_(token_a[0][1], 0.5)
    #
    # # Second token
    # token_b = label_costs[1]
    # eq_(labels[token_b[0][0]], 'A')
    # eq_(token_b[0][1], 0.1)




def test_ignore_ns():
    feat_map = DictFeatMap()

    seqs, labels = read_vw_seq(vw_filename('ns.vw'), feat_map)
    assert b'1^a' in feat_map.feat2index_

    feat_map = DictFeatMap()
    seqs, labels = read_vw_seq(vw_filename('ns.vw'), feat_map, ignore=['1'])
    assert b'1^a' not in feat_map.feat2index_

    # Longer namespaces
    feat_map = DictFeatMap()
    seqs, labels = read_vw_seq(vw_filename('ns.vw'), feat_map, ignore=['3'])
    assert not any(key.startswith(b'3xx') for key in feat_map.feat2index_.keys())

