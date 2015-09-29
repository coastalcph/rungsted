from rungsted.input cimport Example
from rungsted.feat_map cimport FeatMap

cdef class WeightVector:
    cdef:
        public int n
        public tuple dims

        public long n_updates
        public double [::1] w
        public double [::1] acc
        public double [::1] adagrad_squares
        public int [::1] last_update
        public double [::1] active

        public double mean
        public double m2

        public double scaling
        public double decay


        int ada_grad
        int shape0


    cpdef update(self, int feat_i, double val)
    cpdef update2d(self, int i1, int i2, double val)
    cdef inline double get(self, int i1)
    cdef inline double get2d(self, int i1, int i2)
    cdef double score(self, Example *example, int label, FeatMap feat_map)
    cpdef double variance(self)
    cpdef double stddev(self)

    cdef void _update_running_mean(self, double old_val, double new_val)
    cdef void _update_ada_grad(self, int feat_i, double val)
