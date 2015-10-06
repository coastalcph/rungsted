#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: profile=False

from libc.stdio cimport *
from libc.stdint cimport uint32_t, int32_t, int64_t, uint64_t

import random
import cython

from rungsted.feat_map cimport FeatMap
from rungsted.input cimport Example, Feature, Sequence, example_cost, LabelCost
from rungsted.weights cimport WeightVector

import numpy as np

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

    cdef int32_t gold_feat_index = 0
    cdef int32_t pred_feat_index = 0


    # Update emission features
    for word_i in range(len(sent)):
        cur = sent.examples.at(word_i)
        pred_label = cur.pred_label
        gold_label = cur.gold_label

        # Update if prediction is not correct
        if gold_label != pred_label:
            for feat in cur.features:
                gold_feat_index = feat_map.feat_i_for_label(feat.index, gold_label)
                pred_feat_index = feat_map.feat_i_for_label(feat.index, pred_label)
                if gold_feat_index < 0 or pred_feat_index < 0:
                    print(feat.index, gold_label, pred_label, gold_feat_index, pred_feat_index)

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


def update_weights_confusion(Sequence sent, WeightVector transition, WeightVector emission, double alpha, int n_labels,
                   FeatMap feat_map, double[:, :] confusions):
    cdef int word_i, i
    cdef Example cur, prev
    cdef int pred_label, gold_label
    cdef Feature feat
    cdef double update_scaling = 1

    # Update emission features
    for word_i in range(len(sent)):
        cur = sent.examples.at(word_i)
        pred_label = cur.pred_label
        gold_label = cur.gold_label

        # Update if prediction is not correct
        if gold_label != pred_label:
            update_scaling = confusions[gold_label, pred_label] * alpha * cur.importance
            for feat in cur.features:
                new_feat_i = feat_map.feat_i_for_label(feat.index, gold_label)

                emission.update(feat_map.feat_i_for_label(feat.index, gold_label),
                           feat.value * update_scaling)
                emission.update(feat_map.feat_i_for_label(feat.index, pred_label),
                           -feat.value * update_scaling )

            # Transition from from initial state
            if word_i == 0:
                transition.update2d(n_labels, gold_label, update_scaling)
                transition.update2d(n_labels, pred_label, -update_scaling)

    # Transition features
    for word_i in range(1, len(sent)):
        cur = sent.examples.at(word_i)
        prev = sent.examples.at(word_i - 1)

        # If any of the current or previous predictions are incorrect
        if cur.gold_label != cur.pred_label or prev.gold_label != prev.pred_label:

            # If both current and previous prediction are incorrect
            if cur.gold_label != cur.pred_label and prev.gold_label != prev.pred_label:
                update_scaling = alpha
                update_scaling *= (confusions[cur.gold_label, cur.pred_label] + confusions[prev.gold_label, prev.pred_label]) / 2
                update_scaling *= (cur.importance + prev.importance) / 2
            # If only current prediction is incorrect
            elif cur.gold_label != cur.pred_label:
                update_scaling = confusions[cur.gold_label, cur.pred_label] * alpha * cur.importance
            # If only previous prediction is incorrect
            elif prev.gold_label != prev.pred_label:
                update_scaling = confusions[prev.gold_label, prev.pred_label] * alpha * cur.importance

            transition.update2d(cur.gold_label, prev.gold_label, update_scaling)
            transition.update2d(cur.pred_label, prev.pred_label, -update_scaling)





def update_weights_cs_sample(Sequence sent, WeightVector transition, WeightVector emission, double alpha, int n_labels,
                      FeatMap feat_map):
    cdef:
        int word_i, i
        Example cur, prev
        int label
        Feature feat
        double pred_cost, cost, total_inv_cost
        LabelCost label_cost, chosen_label


    # Update emission features
    for word_i in range(len(sent)):
        cur = sent.examples[word_i]
        pred_cost = example_cost(cur, cur.pred_label)

        if pred_cost > .0:
            sample_p = np.array([pred_cost - label_cost.cost for label_cost in cur.labels])
            sample_p[sample_p < 0] = 0
            sample_p_sum = sample_p.sum()
            if sample_p_sum <= 0:
                continue

            sample_p /= sample_p_sum
            chosen_label_index = np.random.choice(np.arange(len(cur.labels)), p=sample_p)
            chosen_label = cur.labels[chosen_label_index]
            # print sample_p, chosen_label_index

            # Negative update
            for feat in cur.features:
                emission.update(feat_map.feat_i_for_label(feat.index, cur.pred_label), -feat.value * alpha * pred_cost)


            # Positive update
            for sample_i, p in enumerate(sample_p):
                if p > 0:
                    for feat in cur.features:
                        emission.update(feat_map.feat_i_for_label(feat.index, cur.labels[sample_i].label), feat.value * alpha * p * (pred_cost - cur.labels[sample_i].cost))


            # # Positive update
            # for feat in cur.features:
            #     # TODO scale by difference between the `pred_cost` and the `chosen_label.cost`
            #     emission.update(feat_map.feat_i_for_label(feat.index, chosen_label.label), feat.value * alpha * (pred_cost - chosen_label.cost))



    update_transition_cs_sample(sent, transition, alpha, n_labels)


    #@cython.cdivision(True)
cdef update_transition_cs_sample(Sequence sent, WeightVector transition, double alpha, int n_labels):
    cdef:
        int word_i
        Example cur, prev
        double bigram_pred_cost, bigram_cost, total_bigram_inv_cost, this_bigram_factor
        int label_cur, label_prev


    # Transition from start state
    if len(sent) > 0:
        cur = sent.examples[0]
        if cur.pred_cost > 0:
            pass
            # transition.update2d(n_labels, cur.pred_label - 1, -alpha * cur.pred_cost)
            # TODO positive update

            #transition.update2d(n_labels, cur.gold_label - 1, alpha * cur.pred_cost)


    # Internal transitions
    for word_i in range(1, len(sent)):
        cur = sent.examples[word_i]
        prev = sent.examples[word_i - 1]

        bigram_pred_cost = (example_cost(cur, cur.pred_label) + example_cost(prev, prev.pred_label))

        # If cost of current or previous prediction is not zero
        if bigram_pred_cost > 0:
            samples = [(cur_label_cost.label, prev_label_cost.label, bigram_pred_cost - (cur_label_cost.cost + prev_label_cost.cost))
                       for cur_label_cost in cur.labels for prev_label_cost in prev.labels]

            sample_p = np.array([tup[2] for tup in samples])
            sample_p[sample_p < 0] = 0
            sum_of_sample_p = sample_p.sum()
            if sample_p.sum() <= 0:
                continue

            sample_p /= sum_of_sample_p

            # print "bigram_pred_cost", bigram_pred_cost
            # print samples, sample_p

            chosen_sample_index = np.random.choice(np.arange(len(samples)), p=sample_p)

            cur_label = samples[chosen_sample_index][0]
            prev_label = samples[chosen_sample_index][1]

            # Negative update
            transition.update2d(cur.pred_label, prev.pred_label, -alpha * bigram_pred_cost)

            # Positive update
            transition.update2d(cur_label, prev_label, alpha * bigram_pred_cost)

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
