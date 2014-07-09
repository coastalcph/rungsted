#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: profile=True

import cython
import math
import numpy as np
#cimport numpy as np

from feat_map cimport FeatMap
from input cimport Example, Sequence, Feature, example_cost
from struct_perceptron cimport e_score

from weights cimport WeightVector

cdef extern from "math.h":
    float INFINITY
    double exp(double)
    double log(double)


cdef class Viterbi(object):
    cdef:
        int n_labels
        WeightVector transition
        WeightVector emission
        FeatMap feat_map
        double[:, ::1] trellis
        int[:, ::1] path

    def __init__(self, int n_labels, WeightVector transition, WeightVector emission, FeatMap feat_map):
        self.n_labels = n_labels
        self.emission = emission
        self.transition = transition
        self.feat_map = feat_map
        self._alloc_trellis(50)


    cdef _alloc_trellis(self, int size):
        self.trellis = np.zeros((size, self.n_labels), dtype=np.float64)
        # Allocate back pointers
        self.path = np.zeros((size, self.n_labels), dtype=np.int32)*-1

    cdef _constrain_labels(self, Sequence sent):
        cdef int word = 0
        for e in sent.examples:
            if e.constraints.size() > 0:
                for label in range(self.n_labels):
                    self.trellis[word, label] = -INFINITY
                for label in e.constraints:
                    self.trellis[word, label] = 0
            word += 1

    def decode(self, Sequence sent):
        if self.trellis.shape[0] < len(sent):
            self._alloc_trellis(int(math.ceil(len(sent) * 1.1)))
        # self._alloc_trellis(len(sent))

        self._constrain_labels(sent)
        self._fill_trellis(sent)
        best_seq = self._recover_best_seq(len(sent))
        self._update_predictions(sent, best_seq)

        return best_seq

    cdef _update_predictions(self, Sequence sent, list best_seq):
        cdef:
            int pred_label

        for i in range(sent.examples.size()):
            pred_label = best_seq[i]
            sent.examples[i].pred_label = pred_label
            sent.examples[i].pred_cost = example_cost(sent.examples[i], pred_label)

    @cython.wraparound(True)
    cdef list _recover_best_seq(self, int seq_len):
        best_seq = [np.asarray(self.trellis)[seq_len-1].argmax()]
        for i in reversed(range(1, seq_len)):
            best_seq.append(self.path[i, <int> best_seq[-1]])
        return [label for label in reversed(best_seq)]


    cdef void _fill_trellis(self, Sequence seq):
        cdef:
            double min_score
            double score
            int min_prev
            double e_score

            int cur_label, prev_label
            int  feat_i, i, j
            int word_i = 0
            double feat_val
            Example cur
            Feature feat

            int offset

        for word_i in range(seq.examples.size()):
            cur = seq.examples[word_i]
            # Current label
            for cur_label in range(self.n_labels):
                if self.trellis[word_i, cur_label] == -INFINITY:
                    continue

                min_score = -1E9
                min_prev = -1

                # Emission score
                # NOTE: 10% faster if passing section size instead of self.feat_map
                e_score = self.emission.score(&cur, cur_label, self.feat_map)

                # Previous label
                # Transitions from start state
                if word_i == 0:
                    self.trellis[word_i, cur_label] = e_score + self.transition.get2d(self.n_labels, cur_label)
                    self.path[word_i, cur_label] = cur_label
                # Transitions from the rest of the states
                else:
                    offset = cur_label * (self.n_labels + 2) # Including labels for initial and final states
                    for prev_label in range(self.n_labels):
                        score = e_score + self.transition.w[offset + prev_label] + self.trellis[word_i-1, prev_label]

                        if score >= min_score:
                            min_score = score
                            min_prev = prev_label
                    self.trellis[word_i, cur_label] = min_score
                    self.path[word_i, cur_label] = min_prev

# cdef class LogViterbi(object):
#     cdef:
#         int n_labels
#         Weights w
#         FeatMap feat_map
#         double[:, ::1] trellis
#         int[:, ::1] path
#         double[::1] score_by_label
#
#     def __init__(self, int n_labels, Weights w, FeatMap feat_map):
#         self.n_labels = n_labels
#         self.w = w
#         self.feat_map = feat_map
#         self._alloc_trellis(50)
#
#
#     cdef _alloc_trellis(self, int size):
#         self.trellis = np.zeros((size, self.n_labels), dtype=np.float64)
#         # Allocate back pointers
#         self.path = np.zeros((size, self.n_labels), dtype=np.int32)*-1
#         self.score_by_label = np.zeros(self.n_labels)
#
#     cdef _constrain_labels(self, Sequence sent):
#         cdef int word = 0
#         for e in sent.examples:
#             if e.constraints.size() > 0:
#                 for label in range(self.n_labels):
#                     self.trellis[word, label] = -INFINITY
#                 for label in e.constraints:
#                     self.trellis[word, label] = 0
#             word += 1
#
#     def decode(self, Sequence sent):
#         if self.trellis.shape[0] < len(sent):
#             self._alloc_trellis(int(math.ceil(len(sent) * 1.1)))
#         # self._alloc_trellis(len(sent))
#
#         self._constrain_labels(sent)
#         self._fill_trellis(sent)
#         best_seq = self._recover_best_seq(len(sent))
#         self._update_predictions(sent, best_seq)
#
#         return best_seq
#
#     cdef _update_predictions(self, Sequence sent, list best_seq):
#         cdef:
#             int pred_label
#
#         for i in range(sent.examples.size()):
#             pred_label = best_seq[i]
#             sent.examples[i].pred_label = pred_label
#             sent.examples[i].pred_cost = example_cost(sent.examples[i], pred_label)
#
#     @cython.wraparound(True)
#     cdef list _recover_best_seq(self, int seq_len):
#         best_seq = [np.asarray(self.trellis)[seq_len-1].argmax()]
#         for i in reversed(range(1, seq_len)):
#             best_seq.append(self.path[i, <int> best_seq[-1]])
#         return [label for label in reversed(best_seq)]
#
#
#     cdef void _fill_trellis(self, Sequence seq):
#         cdef:
#             double min_score
#             double score
#             int min_prev
#             double e_score, z_norm
#
#             int cur_label, prev_label
#             int  feat_i, i, j
#             int word_i = 0
#             double feat_val
#             Example cur
#             Feature feat
#             double[:, ::1] t_norm
#             double e_log_p = 0
#
#         t_np = np.asarray(self.w.t)
#         t_norm = t_np - np.log(np.exp(t_np).sum(axis=0))
#
#         for word_i in range(seq.examples.size()):
#             cur = seq.examples[word_i]
#             # Compute e_score = P(label|token) for all labels
#
#             # Compute normalization for token
#             z_norm = 0
#             for cur_label in range(self.n_labels):
#                 e_score = 0
#                 for feat in cur.features:
#                     feat_i = self.feat_map.feat_i_for_label(feat.index, cur_label)
#                     e_score += self.w.e[feat_i] * feat.value
#                 self.score_by_label[cur_label] = e_score
#                 z_norm += exp(e_score)
#             z_norm = log(z_norm)
#
#             for cur_label in range(self.n_labels):
#                 e_log_p = self.score_by_label[cur_label] - z_norm
#
#                 min_score = -1E9
#                 min_prev = -1
#
#                 # Previous label
#                 # Transitions from start state
#                 if word_i == 0:
#                     self.trellis[word_i, cur_label] = e_log_p + t_norm[self.n_labels, cur_label]
#                     self.path[word_i, cur_label] = cur_label
#                 # Transitions from the rest of the states
#                 else:
#                     for prev_label in range(self.n_labels):
#                         score = e_log_p + t_norm[cur_label, prev_label] + self.trellis[word_i-1, prev_label]
#
#                         if score >= min_score:
#                             min_score = score
#                             min_prev = prev_label
#                     self.trellis[word_i, cur_label] = min_score
#                     self.path[word_i, cur_label] = min_prev
