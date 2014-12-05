import os
from nose.tools import eq_

from rungsted.feat_map import DictFeatMap
from rungsted.input import read_vw_seq



def vw_filename(fname):
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    return os.path.join(data_dir, fname)


def test_cs_weights():
    seqs, labels = read_vw_seq(vw_filename('cs.vw'), DictFeatMap())
    eq_(len(seqs), 1)
    eq_(set(labels), set(['A', 'B', 'C']))

    label_costs = seqs[0].label_costs

    # First token
    token_a = label_costs[0]
    eq_(labels[token_a[0][0]], 'C')
    eq_(token_a[0][1], 0.5)

    # Second token
    token_b = label_costs[1]
    eq_(labels[token_b[0][0]], 'A')
    eq_(token_b[0][1], 0.1)

    eq_(labels[token_b[1][0]], 'B')
    eq_(token_b[1][1], 0.2)

