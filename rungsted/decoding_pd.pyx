#cython: boundscheck=False
#cython: nonecheck=False
#cython: profile=True
import IPython

import cython
import math
import numpy as np
cimport numpy as cnp

from operator import mul
import sys

from feat_map cimport FeatMap
from input cimport Example, Sequence, Feature, example_cost
from pydecode import ChartBuilder
import pydecode


from weights cimport WeightVector

cdef extern from "math.h":
    float INFINITY
    double exp(double)
    double log(double)


# cdef class Viterbi(object):
#     cdef:
#         int n_labels
#         Weights w
#         FeatMap feat_map
#         dict chart_cache
#         int START_LABEL
#         int FINAL_LABEL
#
#     def __init__(self, int n_labels, Weights w, FeatMap feat_map):
#         self.n_labels = n_labels
#         self.w = w
#         self.feat_map = feat_map
#         self.chart_cache = {}
#         self.START_LABEL = n_labels
#         self.FINAL_LABEL = n_labels +1
#
#     cdef _build_chart(self, int n_tokens):
#         if n_tokens in self.chart_cache:
#             return self.chart_cache[n_tokens]
#
#         assert n_tokens >= 1
#
#         # Items in the HMM are arranged in a lattice structure with a row
#         # for each token in the input plus rows for the initial and final states.
#         # Columns correspond to tags, except in the initial and final states.
#         n_rows = n_tokens + 2
#         n_cols = self.n_labels + 2
#         items_dims = (n_rows, n_cols)
#         items = np.arange(reduce(mul, items_dims)).reshape(items_dims)
#
#         # Mark invalid items for start and final state
#         items[0, :self.START_LABEL] = -1
#         items[0, self.FINAL_LABEL] = -1
#         items[-1, :self.FINAL_LABEL] = -1
#
#         # Outputs represent the edges in the lattice.
#         # The three dimensions are: token, cur_tag, prev_tag
#         output_dims = (n_rows - 1, n_cols, n_cols)
#         outputs = np.arange(reduce(mul, output_dims)).reshape(output_dims)
#
#         c = ChartBuilder(items, outputs)
#         # Terminal nodes
#         c.init(items[0, [self.START_LABEL]])
#
#         # First token to start state
#         cdef int i, j
#         for j in range(self.n_labels):
#             # The terminal item depends on each item in the first row
#             c.set(items[1, j], items[0, [self.START_LABEL]], out=outputs[0, j, [self.START_LABEL]])
#
#         # Between tokens
#         for i in range(2, n_rows-1):
#             for j in range(self.n_labels):
#                 c.set(items[i, j], items[i-1, :-2], out=outputs[i-1, j, :-2])
#
#         # Final state to last token
#         c.set(items[-1, self.FINAL_LABEL], items[-2, :-2], out=outputs[-1, self.FINAL_LABEL, :-2])
#
#         self.chart_cache[n_tokens] = (c, c.finish())
#
#         return self.chart_cache[n_tokens]
#         # return c, c.finish()
#
#
#     @cython.wraparound(False)
#     cdef double[:, :, ::1] _make_scores(self, Sequence sent, outputs):
#         # The i-th column in output_scores contains all hypergraph edges from the i+1-th to the i-th layer.
#         # Indices along the second and third dimensions refer to the label of the i-th layer and i+1-th layer.
#         cdef double[:, :, ::1] output_scores
#         output_scores = np.zeros(outputs.shape)
#
#         cdef:
#             int i = 0, j = 0
#             Example cur
#             double e_score, t_score
#
#         # Dependencies between first token and start state
#         for j in range(self.n_labels):
#             output_scores[0, j, self.START_LABEL] = self.w.t[j, self.START_LABEL]
#
#         # Dependencies between tokens
#         for i in range(sent.examples.size()):
#             cur = sent.examples[i]
#             for cur_label in range(self.n_labels):
#                 e_score = self._e_score(cur, cur_label)
#
#                 for next_label in range(self.n_labels):
#                     t_score = self.w.t[next_label, cur_label]
#                     output_scores[i + 1, next_label, cur_label] = e_score + t_score
#
#
#         # Dependencies between final state and last token
#         for j in range(self.n_labels):
#             e_score = self._e_score(sent.examples.back(), j)
#             t_score = self.w.t[self.FINAL_LABEL, j]
#             output_scores[sent.examples.size(), self.FINAL_LABEL, j] = e_score + t_score
#
#
#         return output_scores
#
#     cdef double _e_score(self, Example example, int label):
#         cdef double e_score = 0
#         for feat in example.features:
#             feat_i = self.feat_map.feat_i_for_label(feat.index, label)
#             e_score += self.w.e[feat_i] * feat.value
#         return e_score
#
#
#     def decode(self, Sequence sent):
#         chart, dp = self._build_chart(sent.examples.size())
#         scores = self._make_scores(sent, dp.outputs)
#
#         best = pydecode.argmax(dp, np.asarray(scores))
#
#         best_labels = [cur_i for i, next_i, cur_i in best][1:]
#         self._update_predictions(sent, best_labels)
#
#         return best_labels
#
#     cdef _update_predictions(self, Sequence sent, list best_seq):
#         assert len(sent) == len(best_seq)
#
#         cdef:
#             int pred_label
#
#         for i in range(sent.examples.size()):
#             pred_label = best_seq[i]
#             sent.examples[i].pred_label = pred_label
#             sent.examples[i].pred_cost = example_cost(sent.examples[i], pred_label)
