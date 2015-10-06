#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: profile=False
#cython: cdivision=True

from libc.stdio cimport *
from libc.math cimport isnan
import numpy as np
cimport numpy as cnp

cdef extern from "math.h":
    double sqrt(double)
    double log(double)
    double pow(double, double)

cdef extern from "<cmath>" namespace "std":
    int isnormal(double)


cnp.import_array()

cdef class WeightVector:
    def __init__(self, dims, ada_grad=True, w=None, l2_decay=None):
        if isinstance(dims, int):
            self.n = dims
            self.dims = (dims,)
        elif isinstance(dims, tuple):
            self.n = 1
            for dim in dims:
                self.n *= dim
            self.dims = dims
            self.shape0 = dims[0]

        if w is None:
            self.w = np.zeros(self.n, dtype=np.float64)
        else:
            assert w.size == self.n
            self.w = w

        self.ada_grad = int(ada_grad)
        self.acc = np.zeros_like(self.w, dtype=np.float64)
        self.base = np.zeros_like(self.w, dtype=np.float64)
        self.adagrad_squares = np.ones_like(self.w, dtype=np.float64)
        self.last_update = np.zeros_like(self.w, dtype=np.int32)
        self.active = np.ones_like(self.w, dtype=np.float64)
        self.decay = 1 - l2_decay if l2_decay else 1
        self.scaling = 1
        self.n_updates = 0

    def average(self):
        w_copy = np.asarray(self.w)
        acc_copy = np.asarray(self.acc)
        last_update_copy = np.asarray(self.last_update)

        acc_copy += w_copy * (self.n_updates - last_update_copy)
        w_copy = acc_copy / self.n_updates
        self.w = w_copy

    cdef void _update_running_mean(self, double old_val, double new_val):
        # Keep running mean and variance using a variant of Welford's algorithm.
        # This function substitutes `old_val` for `new_val` in the set of
        # numbers the mean and variance are computed over.
        delta = new_val - old_val
        d_old = old_val - self.mean
        self.mean += delta / self.n
        d_new = new_val - self.mean
        self.m2 += delta * (d_old + d_new)

    cdef void _update_ada_grad(self, int feat_i, double val):
        cdef double learning_rate = 1.0
        # print feat_i, self.adagrad_squares[feat_i], sqrt(self.adagrad_squares[feat_i])
        val *= (learning_rate / sqrt(self.adagrad_squares[feat_i]))
        self.adagrad_squares[feat_i] += val*val

    cpdef void update_done(self):
        self.n_updates += 1


    cdef update(self, int feat_i, double val):
        if feat_i < 0:
            raise ValueError("feature index is < 0")

        val *= self.active[feat_i]
        if val == 0:
            return

        cdef int missed_updates = self.n_updates - self.last_update[feat_i] - 1

        # Perform missing updates for previous rounds
        self.last_update[feat_i] = self.n_updates
        self.acc[feat_i] += (missed_updates + 1) * self.w[feat_i]

        self._update_running_mean(self.w[feat_i], self.w[feat_i] + val)
        if self.ada_grad:
            self._update_ada_grad(feat_i, val)

        # New update
        self.acc[feat_i] += val
        self.w[feat_i] += val


    cpdef double variance(self):
        return self.m2 / float(self.n)

    cpdef double stddev(self):
        return sqrt(self.variance())

    cpdef update2d(self, int i1, int i2, double val):
        self.update(self.shape0 * i1 + i2, val)

    cdef double get(self, int i1):
        return self.w[i1]

    cdef double get2d(self, int i1, int i2):
        return self.w[self.shape0 * i1 + i2]


    cdef double score(self, Example *example, int label, FeatMap feat_map):
        cdef double e_score = 0
        cdef int feat_i
        for feat in example.features:
            feat_i = feat_map.feat_i_for_label(feat.index, label)
            e_score += self.w[feat_i] * feat.value * self.active[feat_i]
        return e_score


    cpdef void rescale(self):
        self.scaling = 1


    @classmethod
    def load(cls, file, **kwargs):
        with np.load(file) as npz_file:
            dims = tuple(npz_file['dims'])
            w = WeightVector(dims, **kwargs)
            w.ada_grad = npz_file['ada_grad'].sum()
            w.w = npz_file['w']
            w.acc = npz_file['acc']
            w.adagrad_squares = npz_file['adagrad_squares']
            w.last_update = npz_file['last_update']
            w.active = np.ones_like(w.w, dtype=np.float64)

            return w

    def save(self, file):
        np.savez(file,
                 ada_grad=self.ada_grad,
                 w=self.w,
                 acc=self.acc,
                 adagrad_squares=self.adagrad_squares,
                 last_update=self.last_update,
                 dims=self.dims)

    def copy(self):
        return WeightVector(self.dims, self.ada_grad, np.asarray(self.w).copy())

cdef class ScaledWeightVector(WeightVector):
    cdef update(self, int feat_i, double val):
        # The parameter `val` is not scaled
        # `self.acc` is not affected by the scaling

        # cdef double orig_val = val
        if feat_i < 0:
            raise ValueError("feature index is < 0 (actual: {})".format(feat_i))

        if not self.active[feat_i]:
            return

        # Perform missing updates for previous rounds.
        cdef int missed_updates = self.n_updates - self.last_update[feat_i] - 1
        self.last_update[feat_i] = self.n_updates

        # Without scaling, the update to the cumulative vector would simply be
        #
        #    missed_updates * self.w[feat_i]
        #
        # However, we have to adjust for the fact that the weights decay.
        # It's convenient to start at the current (scaled) value and calculate
        # how much larger each of the missed updates should be.
        # If the current time is t=0, d is the decay factor (e.g. 0.99), and v is the
        # current value of the weight, the update at any past timestamp (t < 0) is:
        #
        #    u(t) = d^t * v
        #
        # The total update for N missed updates is therefore
        #
        #   \sum_{t=1}^{N} u(t) = \sum_{t=1}^{N} d^t * v = (\sum_{t=1}^{N} d^t) * v

        # To avoid explicitly calculating all terms in the sum, we use the fact that
        #
        #    $\sum_{i=0}^{N-1} r^i = \frac{1-r^N}{1 - r}$

        cdef float backward_scaling_factor = 1 - self.decay + 1
        cdef double scaling_factor_missing = ((1 - pow(backward_scaling_factor, missed_updates + 2))
                                              / (1 - backward_scaling_factor)) - 1

        if isnormal(scaling_factor_missing):
            self.acc[feat_i] += scaling_factor_missing * self.w[feat_i] * self.scaling
            if isnan(self.acc[feat_i]):
                print("BAD VAL", self.w[feat_i], scaling_factor_missing, self.scaling, missed_updates)
                exit(0)


        # New update
        self.acc[feat_i] += val
        self.w[feat_i] += val * (1.0 / self.scaling)

    # Subclasssing prevents inlining
    cdef double get(self, int i1):
        return self.base[i1] + self.w[i1] * self.scaling

    cdef double get2d(self, int i1, int i2):
        cdef int index = self.shape0 * i1 + i2
        return self.base[index] + self.w[index] * self.scaling

    cpdef void update_done(self):
        self.n_updates += 1
        self.scaling *= self.decay
        if self.scaling < 0.000000001:
            self.rescale()

    cpdef void rescale(self):
        cdef int i
        for i in range(len(self.w)):
            self.w[i] *= self.scaling

        self.scaling = 1.0


    def average(self):
        # Catch up with missing updates
        # cdef double scaling_factor_missing = ((1 - pow(self.decay, missed_updates + 1)) / (1 - self.decay)) - 1
        cdef float backward_scaling_factor = 1 - self.decay + 1
        cdef int i
        cdef double scaling_factor_missing
        cdef int missed_updates

        for i in range(len(self.w)):
            missed_updates = self.n_updates - self.last_update[i] - 1
            scaling_factor_missing = ((1 - pow(backward_scaling_factor, missed_updates + 2)) / (1 - backward_scaling_factor)) - 1
            self.last_update[i] = self.n_updates
            if isnormal(scaling_factor_missing):
                self.acc[i] += scaling_factor_missing * self.w[i] * self.scaling

        self.w = np.asarray(self.acc) / self.n_updates


    cdef double score(self, Example *example, int label, FeatMap feat_map):
        cdef double e_score = 0
        cdef int feat_i
        for feat in example.features:
            feat_i = feat_map.feat_i_for_label(feat.index, label)
            e_score += self.base[feat_i] + self.w[feat_i] * self.scaling * feat.value *  self.active[feat_i]
        return e_score

    def save(self, file):
        self.w = np.asarray(self.base) + np.array(self.w)
        super().save(file)

    @classmethod
    def load(cls, file, **kwargs):
        with np.load(file) as npz_file:
            dims = tuple(npz_file['dims'])
            w = ScaledWeightVector(dims, **kwargs)
            w.ada_grad = npz_file['ada_grad'].sum()
            w.w = npz_file['w']
            w.acc = npz_file['acc']
            w.adagrad_squares = npz_file['adagrad_squares']
            w.last_update = npz_file['last_update']
            w.active = np.ones_like(w.w, dtype=np.float64)

            return w
