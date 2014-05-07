from libcpp.vector cimport vector

cdef class Dataset(object):
    cdef:
        char *quadratic
        char *ignore
        int nnz
        int n_labels


cdef struct s_feature:
    int index
    double value

ctypedef s_feature Feature

cdef class Example(object):
    cdef:
        char * id_
        Dataset dataset
        double importance
        public double[:] cost
        vector[Feature] features
        vector[int] constraints

    cpdef int flat_label(self)
    cdef inline int add_feature(self, int, double)
