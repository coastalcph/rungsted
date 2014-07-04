# coding: utf-8
import argparse
from collections import defaultdict
import logging
import os
import random
import cPickle
import numpy as np
import sys
from os.path import exists
import time
from decoding import Viterbi, LogViterbi
from feat_map import HashingFeatMap, DictFeatMap

from input import read_vw_seq
from timer import Timer
# from struct_perceptron import Weights, viterbi, update_weights, avg_loss, accuracy, update_weights_cs
from struct_perceptron import Weights, viterbi, update_weights, avg_loss, accuracy

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(description="""Structured perceptron tagger""")
parser.add_argument('--train', help="Training data (vw format)")
parser.add_argument('--test', help="Test data (vw format)")
parser.add_argument('--hash-bits', '-b', help="Size of feature vector in bits (2**b)", type=int)
parser.add_argument('--passes', help="Number of passes over the training set", type=int, default=5)
parser.add_argument('--predictions', '-p', help="File for outputting predictions")
parser.add_argument('--ignore', help="One-character prefix of namespaces to ignore", nargs='*', default=[])
parser.add_argument('--quadratic', '-q', help="Combine features in these two namespace, identified by a one-character prefix of their name"
                                              "':' is a short-hand for all namespaces", nargs='*', default=[])
parser.add_argument('--shuffle', help="Shuffle examples after each iteration", action='store_true')
parser.add_argument('--no-average', help="Do not average over all updates", action='store_true')
parser.add_argument('--initial-model', '-i', help="Initial model from this file")
parser.add_argument('--final-model', '-f', help="Save model here after training")
parser.add_argument('--cost-sensitive', '--cs', help="Cost-sensitive weight updates", action='store_true')
parser.add_argument('--audit', help="Print the interpretation of the input files to standard out. "
                                    "Useful for debugging", action='store_true')

args = parser.parse_args()


timers = defaultdict(lambda: Timer())
logging.info("Tagger started. \nCalled with {}".format(args))

if args.hash_bits:
    feat_map = HashingFeatMap(args.hash_bits)
else:
    feat_map = DictFeatMap()

# weight_updater = update_weights_cs if args.cost_sensitive else update_weights
weight_updater = update_weights

if args.initial_model:
    if not args.hash_bits and exists(args.initial_model + ".features"):
        feat_map.feat2index_ = cPickle.load(open(args.initial_model + ".features"))

train = None
train_labels = None
if args.train:
    train, train_labels = read_vw_seq(args.train, ignore=args.ignore, quadratic=args.quadratic, feat_map=feat_map,
                                      audit=args.audit)
    logging.info("Training data {} sentences {} labels".format(len(train), len(train_labels)))

# Prevents the addition of new features when loading the test set
feat_map.freeze()
test = None
if args.test:
    # FIXME labels may be loaded from saved file
    test, test_labels = read_vw_seq(args.test, ignore=args.ignore, quadratic=args.quadratic, feat_map=feat_map,
                                    labels=train_labels, audit=args.audit)
    logging.info("Test data {} sentences {} labels".format(len(test), len(test_labels)))
    assert len(train_labels) == len(test_labels)

# FIXME labels may be loaded from saved file
n_labels = len(train_labels)
if not args.hash_bits:
    feat_map.n_labels = n_labels


# Loading weights
w = Weights(n_labels, feat_map.n_feats())
logging.info("Weight vector size {}".format(feat_map.n_feats()))
if args.initial_model:
    w.load(open(args.initial_model))

if not args.hash_bits and args.final_model:
    cPickle.dump(feat_map.feat2index_, open(args.final_model + ".features", 'w'), protocol=2)

n_updates = 0

vit = LogViterbi(n_labels, w, feat_map)

# Training loop
if args.train:
    timers['train'].begin()
    epoch_msg = ""
    for epoch in range(1, args.passes+1):
        learning_rate = 0.1
        if args.shuffle:
            random.shuffle(train)
        for sent in train:
            vit.decode(sent)
            weight_updater(sent, w, learning_rate, n_labels, feat_map)
            w.incr_n_updates()

            if w.n_updates % 1000 == 0:
                print >>sys.stderr, '\r[{}] {}k sentences total'.format(epoch, w.n_updates / 1000),

        epoch_msg = "[{}] train loss={:.4f} ".format(epoch, avg_loss(train))
        print >>sys.stderr, "\r{}{}".format(epoch_msg, " "*72)

    timers['train'].end()
    if not args.no_average:
        w.average_weights()

    print >>sys.stderr, "Training took {:.2f} secs. {} words/sec".format(timers['train'].elapsed(),
                                                                         int(sum(len(seq) for seq in train) / timers['train'].elapsed()))

# Testing
if args.test:
    timers['test'].begin()
    with open(args.predictions or os.devnull, 'w') as out:
        labels_map = dict((i, label_str) for i, label_str in enumerate(test_labels))
        for sent_i, sent in enumerate(test):
            if sent_i > 0:
                print >>out, ""
            vit.decode(sent)
            for example_id, gold_label, pred_label in zip(sent.ids, sent.gold_labels, sent.pred_labels):
                print >>out, "{}\t{}\t{}".format(example_id, labels_map[gold_label], labels_map[pred_label])


    timers['test'].end()
    logging.info("Accuracy: {:.3f}".format(accuracy(test)))
    print >>sys.stderr, "Test took {:.2f} secs. {} words/sec".format(timers['test'].elapsed(),
                                                                         int(sum(len(seq) for seq in test) / timers['test'].elapsed()))


# Save model
if args.final_model:
    w.save(open(args.final_model, 'w'))