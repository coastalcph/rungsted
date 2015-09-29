#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

from rungsted.feat_map cimport FeatMap
from rungsted.input cimport Example, Sequence
from rungsted.weights cimport WeightVector

import numpy as np
# cimport numpy as np

from libc.stdint cimport uint32_t
from libc.stdlib cimport rand


cdef extern from "limits.h":
    long RAND_MAX

cdef extern from "stdlib.h":
    long random()
    void srandom(unsigned int seed)

cdef class FastBinomialCorruption(object):
    cdef:
        float drop_pct
        FeatMap feat_map
        int n_labels

    def __init__(self, drop_pct, FeatMap feat_map, int n_labels):
        self.drop_pct = drop_pct
        self.feat_map = feat_map
        self.n_labels = n_labels

    cpdef corrupt_sequence(self, Sequence sent, WeightVector emission, WeightVector transition):
        cdef:
            Example *example
            int i, j, label
            int base_feat_i, feat_i
            long threshold_int = <long> ((1.0 - self.drop_pct) * RAND_MAX)
            int selected = 0, total = 0

        # Do a sparse drop-out of the emissions features
        for i in range(sent.examples.size()):
            example = &sent.examples[i]
            for j in range(example.features.size()):
                base_feat_i = (&example.features[j]).index
                for label in range(self.n_labels):
                    feat_i = self.feat_map.feat_i_for_label(base_feat_i, label)
                    emission.active[feat_i] = 0 if random() > threshold_int else 1

        # And a dense drop-out for the transition
        for i in range(transition.active.shape[0]):
            transition.active[i] = 0 if random() > threshold_int else 1

cdef class DistributionCorruption(object):
    cdef:
        FeatMap feat_map
        int n_labels
        sample_fn
        int capacity
        int current
        double [::1] buffer

    def __init__(self, sample_fn, FeatMap feat_map, int n_labels, int capacity=1000000):
        self.sample_fn = sample_fn
        self.capacity = capacity
        self.feat_map = feat_map
        self.n_labels = n_labels
        self.current = -1
        self.buffer = sample_fn(self.capacity)

    # cdef double _draw(self):
    #     if self.recycle:
    #         return self._randomized_draw()
    #     else:
    #         return self._buffered_draw()

    cdef double _draw(self):
        self.current += 1
        # Buffer is used up. Fill up again
        if self.current == self.capacity:
            self.buffer = self.sample_fn(self.capacity)
            self.current = 0
        return self.buffer[self.current]


    cpdef corrupt_sequence(self, Sequence sent, WeightVector emission, WeightVector transition):
        cdef:
            Example *example
            int j, base_feat_i, label, feat_i

        for i in range(sent.examples.size()):
            example = &sent.examples[i]
            for j in range(example.features.size()):
                base_feat_i = (&example.features[j]).index
                for label in range(self.n_labels):
                    feat_i = self.feat_map.feat_i_for_label(base_feat_i, label)
                    emission.active[feat_i] = self._draw()

        # And a dense drop-out for the transition
        for i in range(transition.active.shape[0]):
            transition.active[i] = self._draw()

cdef class RecycledDistributionCorruption(DistributionCorruption):
    cdef:
        double capacity_scale_factor

    def __init__(self, sample_fn, FeatMap feat_map, int n_labels, int capacity=1000000):
        super(RecycledDistributionCorruption, self).__init__(sample_fn, feat_map, n_labels, capacity)
        self.capacity_scale_factor = (self.capacity / float(RAND_MAX))


    cdef double _draw(self):
        cdef int random_index = int(random() * self.capacity_scale_factor)
        return self.buffer[random_index]

cdef class AdversialCorruption(object):
    cdef:
        double drop_pct
        FeatMap feat_map
        int n_labels

    def __init__(self, drop_pct, FeatMap feat_map, int n_labels):
        self.drop_pct = drop_pct
        self.feat_map = feat_map
        self.n_labels = n_labels

    cpdef corrupt_sequence(self, Sequence sent, WeightVector emission, WeightVector transition):
        cdef:
            Example *example
            int i, j, base_feat_i, label, feat_i
            # int threshold_int = int((1.0 - self.drop_pct) * float(INT_MAX))
            long threshold_int = <long> ((1.0 - self.drop_pct) * RAND_MAX)
            double w_stddev
            # int inactive = 0, total = 0

        w_stddev = emission.stddev()

        # if rand() > (0.99 * INT_MAX):
            # print "mean     {}  variance {}".format(emission.mean, emission.variance())
            # print "mean     {} vs {}, diff = {}".format(emission.mean, w_np.mean(), abs(emission.mean-w_np.mean()))
            # print "variance {} vs {}, diff = {}".format(emission.variance(), w_np.var(ddof=1), abs(emission.variance() - w_np.var()))

        for i in range(sent.examples.size()):
            example = &sent.examples[i]
            for j in range(example.features.size()):
                base_feat_i = (&example.features[j]).index
                for label in range(self.n_labels):
                    feat_i = self.feat_map.feat_i_for_label(base_feat_i, label)
                    # total += 1

                    if random() > threshold_int and abs(emission.w[feat_i] - emission.mean) > w_stddev:
                        # inactive += 1
                        emission.active[feat_i] = 0
                    else:
                        emission.active[feat_i] = 1

        # And a dense drop-out for the transition
        w_stddev = transition.stddev()

        for i in range(transition.active.shape[0]):
            # total += 1
            if rand() > threshold_int and abs(transition.w[i] - transition.mean) > w_stddev:
                # inactive += 1
                transition.active[i] = 0
            else:
                transition.active[i] = 1

        # if rand() > (0.99 * INT_MAX):
        # print "inactive/total = {}/{} = {}".format(inactive, total, inactive/float(total))


def inverse_zipfian_sampler(n_samples, s=3):
    return 1.0 / np.random.zipf(s, n_samples)

