# coding: utf-8
import argparse
import logging
import os
import random
import cPickle
import numpy as np
import sys
from os.path import exists
from feat_map import HashingFeatMap, DictFeatMap

from input import read_vw_seq
from struct_perceptron import Weights, viterbi, update_weights, avg_loss, accuracy

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(description="""Structured perceptron tagger""")
parser.add_argument('--train', help="Training data (vw format)")
parser.add_argument('--test', help="Test data (vw format)")
parser.add_argument('--hash-bits', '-b', help="Size of feature vector in bits (2**b)", type=int)
parser.add_argument('--n-labels', '-k', help="Number of different labels", required=True, type=int)
parser.add_argument('--passes', help="Number of passes over the training set", type=int, default=5)
parser.add_argument('--predictions', '-p', help="File for outputting predictions")
parser.add_argument('--ignore', help="One-character prefix of namespaces to ignore", nargs='*', default=[])
parser.add_argument('--quadratic', '-q', help="Combine features in these two namespace, identified by a one-character prefix of their name"
                                              "':' is a short-hand for all namespaces", nargs='*', default=[])
parser.add_argument('--decay-exp', help="Learning rate decay exponent. Learning rate is (iteration no)^decay_exponent",
                    default=0, type=float)
parser.add_argument('--decay-delay', help="Delay decaying the learning rate for this many iterations",
                    default=10, type=int)
parser.add_argument('--shuffle', help="Shuffle examples after each iteration", action='store_true')
parser.add_argument('--average', help="Average over all updates", action='store_true')
parser.add_argument('--initial-model', '-i', help="Initial model from this file")
parser.add_argument('--final-model', '-f', help="Save model here after training")


args = parser.parse_args()

logging.info("Tagger started. \nCalled with {}".format(args))
n_labels = args.n_labels

if args.hash_bits:
    feat_map = HashingFeatMap(args.hash_bits)
else:
    feat_map = DictFeatMap(args.n_labels)

if args.initial_model:
    if not args.hash_bits and exists(args.initial_model + ".features"):
        feat_map.feat2index_ = cPickle.load(open(args.initial_model + ".features"))

train = None
if args.train:
    train = read_vw_seq(args.train, args.n_labels, ignore=args.ignore, quadratic=args.quadratic, feat_map=feat_map)
    logging.info("Training data {} sentences".format(len(train)))

# Prevents the addition of new features when loading the test set
feat_map.freeze()
test = None
if args.test:
    test = read_vw_seq(args.test, args.n_labels, ignore=args.ignore, quadratic=args.quadratic, feat_map=feat_map)
    logging.info("Test data {} sentences".format(len(test)))


# Loading weights
w = Weights(n_labels, feat_map.n_feats())
logging.info("Weight vector size {}".format(feat_map.n_feats()))
if args.initial_model:
    w.load(open(args.initial_model))

if not args.hash_bits and args.final_model:
    cPickle.dump(feat_map.feat2index_, open(args.final_model + ".features", 'w'), protocol=2)

n_updates = 0

# Training loop
if args.train:
    epoch_msg = ""
    for epoch in range(1, args.passes+1):
        learning_rate = 0.1 if epoch < args.decay_delay else epoch**args.decay_exp * 0.1
        if args.shuffle:
            random.shuffle(train)
        for sent in train:
            viterbi(sent, n_labels, w, feat_map)
            update_weights(sent, w, n_updates, learning_rate, n_labels, feat_map)

            n_updates += 1
            if n_updates % 1000 == 0:
                print >>sys.stderr, '\r{}\t{} k sentences total'.format(epoch_msg, n_updates / 1000),

        epoch_msg = "[{}] train loss={:.4f} ".format(epoch, avg_loss(train))
        print >>sys.stderr, "\r{}{}".format(epoch_msg, " "*72)

    if args.average:
        w.average_weights(n_updates)

# Testing
if args.test:
    with open(args.predictions or os.devnull, 'w') as out:
        for sent in test:
            viterbi(sent, n_labels, w, feat_map)
            for example in sent:
                print >>out, "{}\t{}\t{}".format(example.id_, example.gold_label, example.pred_label)
            print >>out, ""

    logging.info("Accuracy: {:.3f}".format(accuracy(test)))

# Save model
if args.final_model:
    w.save(open(args.final_model, 'w'))