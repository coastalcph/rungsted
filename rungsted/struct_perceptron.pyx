#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

from libc.stdio cimport *
from libc.stdint cimport uint32_t, int32_t, int64_t, uint64_t
import cython
import numpy as np
cimport numpy as cnp

from input cimport Example, Dataset, DataBlock
from hashing cimport hash_ints

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

    cpdef public int hash_bits

    def __init__(self, n_labels, n_e_feats, hash_bits):
        self.t = np.zeros((n_labels+1, n_labels), dtype=np.float64)
        self.t_acc = np.zeros_like(self.t, dtype=np.float64)
        self.t_last_update = np.zeros_like(self.t, dtype=np.int32)

        self.e = np.zeros(n_e_feats, dtype=np.float64)
        self.e_acc = np.zeros_like(self.e, dtype=np.float64)
        self.e_last_update = np.zeros_like(self.e, dtype=np.int32)

        self.hash_bits = hash_bits

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


def update_weights(int[:] pred_seq, int[:] gold_seq, list sent, Weights w, int n_updates, double alpha, int n_labels):
    cdef int word_i
    cdef Example cur
    # Update emission features

    for word_i in range(len(pred_seq)):
        cur = sent[word_i]
        pred_label = pred_seq[word_i]
        gold_label = gold_seq[word_i]

        # Update if prediction is not correct
        if gold_label != pred_label:
            for i in range(cur.index.shape[0]):
                w.update_e(hash_ints(cur.index[i], gold_label, w.hash_bits),
                           cur.val[i] * alpha, n_updates)
                w.update_e(hash_ints(cur.index[i], pred_label, w.hash_bits),
                           -cur.val[i] * alpha, n_updates)

            # Transition from from initial state
            if word_i == 0:
                w.update_t(n_labels, gold_label, alpha, n_updates)
                w.update_t(n_labels, pred_label, -alpha, n_updates)

    # Transition features
    for word_i in range(1, len(pred_seq)):
        # If current or previous prediction is not correct
        if gold_seq[word_i] != pred_seq[word_i] or gold_seq[word_i-1] != pred_seq[word_i-1]:
            w.update_t(gold_seq[word_i], gold_seq[word_i-1], alpha, n_updates)
            w.update_t(pred_seq[word_i], pred_seq[word_i-1], -alpha, n_updates)


@cython.wraparound(True)
def viterbi(list sent, int n_labels, Weights w):
    """Returns best predicted sequence"""
    # Allocate trellis and back pointers
    path = np.zeros((len(sent), n_labels), dtype=np.int32)*-1
    # trellis = sent.allowed_label_matrix(n_labels)
    trellis = np.zeros_like(path, dtype=np.float64)

    viterbi_path(sent, n_labels, w, trellis, path)

    best_seq = [trellis[-1].argmax()]
    for word_i in reversed(range(1, len(path))):
        best_seq.append(path[word_i, best_seq[-1]])

    return list(reversed(best_seq))


cdef viterbi_path(list seq, int n_labels, Weights w, double[:, ::1] trellis, int[:, ::1] path):
    cdef:
        double min_score
        double score
        int min_prev
        double e_score

        int  feat_i, i, j
        int word_i = 0
        double feat_val
        Example cur

    for word_i in range(len(seq)):
        cur = seq[word_i]
        # Current label
        for label_i in range(n_labels):
            if trellis[word_i, label_i] == -INFINITY:
                continue

            min_score = -1E9
            min_prev = -1
            e_score = 0
            # Emission score

            for i in range(cur.index.shape[0]):
                feat_i = hash_ints(cur.index[i], label_i, w.hash_bits)
                e_score += w.e[feat_i] * cur.val[i]

            # Previous label
            if word_i == 0:
                trellis[word_i, label_i] = e_score + w.t[n_labels, label_i]
                path[word_i, label_i] = label_i

            else:
                for label_j in range(n_labels):
                    score = e_score + w.t[label_i, label_j] + trellis[word_i-1, label_j]

                    if score >= min_score:
                        min_score = score
                        min_prev = label_j
                trellis[word_i, label_i] = min_score
                path[word_i, label_i] = min_prev