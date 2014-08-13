#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: profile=False

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

def binomial_drop_out(Sequence sent, WeightVector emission, WeightVector transition, FeatMap feat_map, int n_labels, float drop_pct):
    cdef:
        Example *example
        float rand_num
        int i, j, label
        int base_feat_i, feat_i
        int threshold_int = int((1.0 - drop_pct) * float(INT_MAX))

    # Do a sparse drop-out of the emissions features
    for i in range(sent.examples.size()):
        example = &sent.examples[i]
        for j in range(example.features.size()):
            base_feat_i = (&example.features[j]).index
            for label in range(n_labels):
                feat_i = feat_map.feat_i_for_label(base_feat_i, label)
                emission.active[feat_i] = 0 if rand() > threshold_int else 1

    # And a dense drop-out for the transition
    for i in range(transition.active.shape[0]):
        transition.active[i] = 0 if rand() > threshold_int else 1




cdef double e_score(Example *example, int label, FeatMap feat_map, double[::1] weights) nogil:
    cdef double e_score = 0
    cdef int feat_i

    for feat in example.features:
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
                new_feat_i = feat_map.feat_i_for_label(feat.index, gold_label)

                emission.update(feat_map.feat_i_for_label(feat.index, gold_label),
                           feat.value * alpha * cur.importance)
                emission.update(feat_map.feat_i_for_label(feat.index, pred_label),
                           -feat.value * alpha * cur.importance)

            # Transition from from initial state
            if word_i == 0:
                transition.update2d(n_labels, gold_label, alpha * cur.importance)
                transition.update2d(n_labels, pred_label, -alpha * cur.importance)

    # Transition features
    for word_i in range(1, len(sent)):
        cur = sent.examples.at(word_i)
        prev = sent.examples.at(word_i - 1)
        # If current or previous prediction is not correct
        if cur.gold_label != cur.pred_label or prev.gold_label != prev.pred_label:
            transition.update2d(cur.gold_label, prev.gold_label, alpha * cur.importance)
            transition.update2d(cur.pred_label, prev.pred_label, -alpha * cur.importance)


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
