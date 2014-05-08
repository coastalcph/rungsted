from libcpp.vector cimport vector
from libcpp.string cimport string

cdef class Dataset(object):
    cdef:
        vector[string] quadratic
        int[255] ignore
        int nnz
        int n_labels


cdef struct s_feature:
    int index
    double value

ctypedef s_feature Feature

cdef class Example(object):
    cdef:
        public char * id_
        Dataset dataset
        double importance
        public double[:] cost
        vector[Feature] features
        vector[int] constraints
        public int pred_label
        public int gold_label

    cdef inline int add_feature(self, int, double)
