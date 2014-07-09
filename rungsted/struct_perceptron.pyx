#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: profile=True

from libc.stdio cimport *
from libc.stdint cimport uint32_t, int32_t, int64_t, uint64_t

import random
import cython

from feat_map cimport FeatMap
from .input cimport Example, Feature, Sequence, example_cost
from weights cimport WeightVector

from libc.stdlib cimport rand
cdef extern from "limits.h":
    int INT_MAX



def drop_out(Sequence sent):
    cdef Feature *feat
    cdef Example *example
    cdef float rand_num
    for i in range(sent.examples.size()):
        example = &sent.examples[i]
        for j in range(example.features.size()):
            rand_num = rand() / float(INT_MAX)
            feat = &example.features[j]
            feat.active = 1 if rand_num < 0.9 else 0

cdef double e_score(Example *example, int label, FeatMap feat_map, double[::1] weights) nogil:
    cdef double e_score = 0
    cdef int feat_i

    for feat in example.features:
        if not feat.active:
            continue
        feat_i = feat_map.feat_i_for_label(feat.index, label)
        e_score += weights[feat_i] * feat.value
    return e_score


def update_weights(Sequence sent, WeightVector transition, WeightVector emission, double alpha, int n_labels,
                   FeatMap feat_map):
    cdef int word_i, i
    cdef Example cur, prev
    cdef int pred_label, gold_label
    cdef Feature feat

    # Update emission features
    for word_i in range(len(sent)):
        cur = sent.examples.at(word_i)
        pred_label = cur.pred_label
        gold_label = cur.gold_label

        # Update if prediction is not correct
        if gold_label != pred_label:
            for feat in cur.features:
                if not feat.active: continue
                new_feat_i = feat_map.feat_i_for_label(feat.index, gold_label)
                if new_feat_i < 0:
                    print "below zero -- error", new_feat_i, feat.index, gold_label

                emission.update(feat_map.feat_i_for_label(feat.index, gold_label),
                           feat.value * alpha)
                emission.update(feat_map.feat_i_for_label(feat.index, pred_label),
                           -feat.value * alpha)

            # Transition from from initial state
            if word_i == 0:
                transition.update2d(n_labels, gold_label, alpha)
                transition.update2d(n_labels, pred_label, -alpha)

    # Transition features
    for word_i in range(1, len(sent)):
        cur = sent.examples.at(word_i)
        prev = sent.examples.at(word_i - 1)
        # If current or previous prediction is not correct
        if cur.gold_label != cur.pred_label or prev.gold_label != prev.pred_label:
            transition.update2d(cur.gold_label, prev.gold_label, alpha)
            transition.update2d(cur.pred_label, prev.pred_label, -alpha)


cpdef double avg_loss(list sents):
    cdef:
        Sequence sent
        Example e
        double total_cost = 0
        int n = 0

    for sent in sents:
        for e in sent.examples:
            if e.pred_label >= 0:
                n += 1
                total_cost += example_cost(e, e.pred_label)

    return total_cost / n if n > 0 else 0.0

cpdef double accuracy(list sents):
    cdef:
        Sequence sent
        Example e
        int n = 0, correct = 0

    for sent in sents:
        for e in sent.examples:
            if e.pred_label >= 0:
                n += 1
                if example_cost(e, e.pred_label) == 0.0:
                    correct += 1

    return float(correct) / n if n > 0 else 0.0
