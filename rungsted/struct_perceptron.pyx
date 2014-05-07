#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

from libc.stdio cimport *
from libc.stdint cimport uint32_t, int32_t, int64_t, uint64_t
import cython
import numpy as np
cimport numpy as cnp
from feat_map cimport HashingFeatMap, FeatMap

from .input cimport Example, Dataset, Feature

cdef extern from "math.h":
    float INFINITY

cnp.import_array()

cdef class Weights:
    cpdef public double [:, ::1] t
    cpdef public double [:, ::1] t_acc
    cpdef public int [:, ::1] t_last_update

    cpdef public double [::1] e
    cpdef public double [::1] e_acc
    cpdef public int [::1] e_last_update

    def __init__(self, n_labels, n_e_feats):
        self.t = np.zeros((n_labels+1, n_labels), dtype=np.float64)
        self.t_acc = np.zeros_like(self.t, dtype=np.float64)
        self.t_last_update = np.zeros_like(self.t, dtype=np.int32)

        self.e = np.zeros(n_e_feats, dtype=np.float64)
        self.e_acc = np.zeros_like(self.e, dtype=np.float64)
        self.e_last_update = np.zeros_like(self.e, dtype=np.int32)

    def average_weights(self, n_updates):
        e = np.asarray(self.e)
        e_acc = np.asarray(self.e_acc)
        e_last_update = np.asarray(self.e_last_update)
        t = np.asarray(self.t)
        t_acc = np.asarray(self.t_acc)
        t_last_update = np.asarray(self.t_last_update)

        e_acc += e * (n_updates - e_last_update)
        e = e_acc / n_updates
        self.e = e

        t_acc += t * (n_updates - t_last_update)
        t = t_acc / n_updates
        self.t = t

    def update_e(Weights self, int feat_i, double val, int n_updates):
        cdef int missed_updates = n_updates - self.e_last_update[feat_i] - 1
        with nogil:
            self.e_last_update[feat_i] = n_updates
            self.e_acc[feat_i] += missed_updates * self.e[feat_i]
            self.e_acc[feat_i] += self.e[feat_i] + val
            self.e[feat_i] += val


    def update_t(Weights self, int label_i, int label_j, double val, int n_updates):
        cdef int missed_updates = n_updates - self.t_last_update[label_i, label_j] - 1
        with nogil:
            self.t_last_update[label_i, label_j] = n_updates
            self.t_acc[label_i, label_j] += missed_updates * self.t[label_i, label_j]
            self.t_acc[label_i, label_j] += self.t[label_i, label_j] + val
            self.t[label_i, label_j] += val


def update_weights(int[:] pred_seq, int[:] gold_seq, list sent, Weights w, int n_updates, double alpha, int n_labels,
                   FeatMap feat_map):
    cdef int word_i, i
    cdef Example cur
    cdef int pred_label, gold_label
    cdef Feature feat
    # Update emission features

    for word_i in range(len(pred_seq)):
        cur = sent[word_i]
        pred_label = pred_seq[word_i]
        gold_label = gold_seq[word_i]

        # Update if prediction is not correct
        if gold_label != pred_label:
            for feat in cur.features:
                w.update_e(feat_map.feat_i_for_label(feat.index, gold_label),
                           feat.value * alpha, n_updates)
                w.update_e(feat_map.feat_i_for_label(feat.index, pred_label),
                           -feat.value * alpha, n_updates)

            # Transition from from initial state
            if word_i == 0:
                w.update_t(n_labels, gold_label - 1, alpha, n_updates)
                w.update_t(n_labels, pred_label - 1, -alpha, n_updates)

    # Transition features
    for word_i in range(1, len(pred_seq)):
        # If current or previous prediction is not correct
        if gold_seq[word_i] != pred_seq[word_i] or gold_seq[word_i-1] != pred_seq[word_i-1]:
            w.update_t(gold_seq[word_i] - 1, gold_seq[word_i-1] - 1, alpha, n_updates)
            w.update_t(pred_seq[word_i] - 1, pred_seq[word_i-1] - 1, -alpha, n_updates)


@cython.wraparound(True)
def viterbi(list sent, int n_labels, Weights w, FeatMap feat_map):
    """Returns best predicted sequence"""
    cdef Example e
    cdef Feature feat

    # Allocate trellis and back pointers
    path = np.zeros((len(sent), n_labels), dtype=np.int32)*-1
    # trellis = sent.allowed_label_matrix(n_labels)
    trellis = np.zeros_like(path, dtype=np.float64)

    viterbi_path(sent, n_labels, w, trellis, path, feat_map)

    best_seq = [trellis[-1].argmax()]
    for word_i in reversed(range(1, len(path))):
        best_seq.append(path[word_i, best_seq[-1]])

    return [label + 1 for label in reversed(best_seq)]


cdef viterbi_path(list seq, int n_labels, Weights w, double[:, ::1] trellis, int[:, ::1] path, FeatMap feat_map):
    cdef:
        double min_score
        double score
        int min_prev
        double e_score

        # Zero-based labels
        int cur_label_0, prev_label_0
        int  feat_i, i, j
        int word_i = 0
        double feat_val
        Example cur
        Feature feat

    for word_i in range(len(seq)):
        cur = seq[word_i]
        # Current label
        for cur_label_0 in range(n_labels):
            if trellis[word_i, cur_label_0] == -INFINITY:
                continue

            min_score = -1E9
            min_prev = -1

            # Emission score
            e_score = 0
            for feat in cur.features:
                feat_i = feat_map.feat_i_for_label(feat.index, cur_label_0 + 1)
                e_score += w.e[feat_i] * feat.value

            # Previous label
            if word_i == 0:
                trellis[word_i, cur_label_0] = e_score + w.t[n_labels, cur_label_0]
                path[word_i, cur_label_0] = cur_label_0

            else:
                for prev_label_0 in range(n_labels):
                    score = e_score + w.t[cur_label_0, prev_label_0] + trellis[word_i-1, prev_label_0]

                    if score >= min_score:
                        min_score = score
                        min_prev = prev_label_0
                trellis[word_i, cur_label_0] = min_score
                path[word_i, cur_label_0] = min_prev