# coding: utf-8
import argparse
import logging
import random
import numpy as np
import sys
from feat_map import HashingFeatMap, DictFeatMap

from input import read_vw_seq
from struct_perceptron import Weights, viterbi, update_weights

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(description="""Structured perceptron tagger""")
parser.add_argument('--train', help="Training data (vw format)")
parser.add_argument('--test', help="Test data (vw format)")
parser.add_argument('--hash-bits', '-b', help="Size of feature vector in bits (2**b)", type=int)
parser.add_argument('--n-labels', '-k', help="Number of different labels", required=True, type=int)
parser.add_argument('--passes', help="Number of passes over the training set", type=int, default=5)
parser.add_argument('--predictions', '-p', help="File for outputting predictions")
parser.add_argument('--ignore', help="One-character prefix of namespaces to ignore", nargs='*', default=[])
parser.add_argument('--decay-exp', help="Learning rate decay exponent. Learning rate is (iteration no)^decay_exponent",
                    default=0, type=float)
parser.add_argument('--decay-delay', help="Delay decaying the learning rate for this many iterations",
                    default=10, type=int)
parser.add_argument('--shuffle', help="Shuffle examples after each iteration", action='store_true')
parser.add_argument('--average', help="Average over all updates", action='store_true')


args = parser.parse_args()

logging.info("Tagger started. \nCalled with {}".format(args))
n_labels = args.n_labels

if args.hash_bits:
    feat_map = HashingFeatMap(args.hash_bits)
else:
    feat_map = DictFeatMap(args.n_labels)

train = read_vw_seq(args.train, args.n_labels, ignore=args.ignore, feat_map=feat_map)
logging.info("Training data {} sentences".format(len(train)))
# Prevents the addition of new features when loading the test set
feat_map.freeze()
test = read_vw_seq(args.test, args.n_labels, ignore=args.ignore, feat_map=feat_map)
logging.info("Test data {} sentences".format(len(test)))

w = Weights(n_labels, feat_map.n_feats())
logging.info("Weight vector size {}".format(feat_map.n_feats()))

n_updates = 0


# Learning loop
for epoch in range(1, args.passes+1):
    learning_rate = 0.1 if epoch < args.decay_delay else epoch**args.decay_exp * 0.1
    if args.shuffle:
        random.shuffle(train)
    for sent in train:
        flattened_labels = [e.flat_label() for e in sent]

        gold_seq = np.array(flattened_labels, dtype=np.int32)
        pred_seq = np.array(viterbi(sent, n_labels, w, feat_map), dtype=np.int32)

        assert len(gold_seq) == len(pred_seq)

        update_weights(pred_seq, gold_seq, sent, w, n_updates, learning_rate, n_labels, feat_map)

        n_updates += 1

        if n_updates % 1000 == 0:
            print >>sys.stderr, '\r{} k sentences total'.format(n_updates / 1000),

y_gold = []
y_pred = []

out = None
if args.predictions:
    out = open(args.predictions, 'w')

if args.average:
    w.average_weights(n_updates)

for sent in test:
    y_pred_sent = viterbi(sent, n_labels, w, feat_map)
    y_gold += [e.flat_label() for e in sent]
    y_pred += y_pred_sent

if out:
    out.close()

assert len(y_gold) == len(y_pred)

correct = np.array(y_gold) == np.array(y_pred)

accuracy = correct.sum() / float(len(correct))

print >>sys.stderr, ''
logging.info("Accuracy: {:.3f}".format(accuracy))
