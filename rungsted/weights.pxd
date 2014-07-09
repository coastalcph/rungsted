from input cimport Example
from feat_map cimport FeatMap

cdef class WeightVector:
    cdef:
        public int n
        public tuple dims

        public double [::1] w
        public double [::1] acc
        public double [::1] adagrad_squares
        public int [::1] last_update

        int ada_grad
        int shape0

        public int n_updates


    cpdef update(self, int feat_i, double val)
    cpdef update2d(self, int i1, int i2, double val)
    cdef inline double get(self, int i1)
    cdef inline double get2d(self, int i1, int i2)
    cdef double score(self, Example *example, int label, FeatMap feat_map)
