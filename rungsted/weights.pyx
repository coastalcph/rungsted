#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: profile=False
#cython: cdivision=True

from libc.stdio cimport *
import numpy as np
cimport numpy as cnp

cdef extern from "math.h":
    double sqrt(double)
    double log(double)

cnp.import_array()

cdef class WeightVector:
    def __init__(self, dims, ada_grad=True, w=None):
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
        self.adagrad_squares = np.ones_like(self.w, dtype=np.float64)
        self.last_update = np.zeros_like(self.w, dtype=np.int32)
        self.active = np.ones_like(self.w, dtype=np.float64)

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


    cpdef update(self, int feat_i, double val):
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

    cdef inline double get(self, int i1):
        return self.w[i1]

    cdef inline double get2d(self, int i1, int i2):
        return self.w[self.shape0 * i1 + i2]


    cdef double score(self, Example *example, int label, FeatMap feat_map):
        cdef double e_score = 0
        cdef int feat_i
        for feat in example.features:
            feat_i = feat_map.feat_i_for_label(feat.index, label)
            e_score += self.w[feat_i] * feat.value * self.active[feat_i]
        return e_score


    @classmethod
    def load(cls, file):
        with np.load(file) as npz_file:
            dims = tuple(npz_file['dims'])
            w = WeightVector(dims)
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