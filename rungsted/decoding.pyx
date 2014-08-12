#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: profile=False

import cython
import math
import numpy as np
#cimport numpy as np
import time

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

    cdef _fill_trellis_with_zeros(self, int size):
        cdef int i, j
        for i in range(size):
            for j in range(self.trellis.shape[1]):
                self.trellis[i, j] = 0

    def decode(self, Sequence sent):
        if self.trellis.shape[0] < len(sent):
            self._alloc_trellis(int(math.ceil(len(sent) * 1.1)))
        # self._alloc_trellis(len(sent))
        self._fill_trellis_with_zeros(len(sent))
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


cdef viterbi_fill_trellis(double [:, ::1] e_scores, double [:, ::1] t_scores, double [:, ::1] trellis, int [:, ::1] path):
    cdef:
        double min_score
        double score
        int min_prev
        double e_score

        int cur_label, prev_label
        int  feat_i, i, j
        int word_i = 0
        double feat_val

        int n_words = e_scores.shape[0]
        int n_labels = t_scores.shape[0]

    for word_i in range(n_words):
        # Current label
        for cur_label in range(n_labels):
            min_score = -1E9
            min_prev = -1

            e_score = e_scores[word_i, cur_label]

            # Transitions from start state
            if word_i == 0:
                trellis[word_i, cur_label] = e_score + t_scores[n_labels, cur_label]
                path[word_i, cur_label] = cur_label
            # Transitions from the rest of the states
            else:
                for prev_label in range(n_labels):
                    score = e_score + t_scores[cur_label, prev_label] + trellis[word_i-1, prev_label]

                    if score >= min_score:
                        min_score = score
                        min_prev = prev_label
                trellis[word_i, cur_label] = min_score
                path[word_i, cur_label] = min_prev

def benchmark_viterbi_fill_trellis():
    cdef:
        int n_labels = 12
        int n_words = 50
        int n_rounds = 1000

        # Setup reusable data structures outside benchmark
        double [:, ::1] trellis = np.zeros((n_words, n_labels))
        int [:, ::1] path = np.zeros((n_words, n_labels), dtype=np.int32)

        double [:, ::1] e_scores = np.random.random(n_words * n_labels).reshape((n_words, n_labels))
        double [:, ::1] t_scores = np.random.random(n_labels * n_labels).reshape((n_labels, n_labels))

        int i

    start_time = time.time()
    for i in range(n_rounds):
        # Randomizing the scores is really the slow part
        # e_scores = np.random.random(n_words * n_labels).reshape((n_words, n_labels))
        # t_scores = np.random.random(n_labels * n_labels).reshape((n_labels, n_labels))
        viterbi_fill_trellis(e_scores, t_scores, trellis, path)
    elapsed = time.time() - start_time
    tokens_per_sec = int((n_rounds * n_words) / elapsed)
    print "{:.2f} secs elapsed; {} tokens/sec".format(elapsed, tokens_per_sec)







