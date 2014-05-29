from libcpp.vector cimport vector
from libcpp.string cimport string

cdef struct dataset_s:
    vector[string] quadratic
    int * ignore
    int nnz
    int n_labels

ctypedef dataset_s Dataset

cdef struct s_feature:
    int index
    double value

ctypedef s_feature Feature

cdef struct example_s:
    Dataset dataset
    char * id_
    double importance
    double * cost
    vector[Feature] features
    vector[int] constraints
    int pred_label
    int gold_label
    double pred_cost

ctypedef example_s Example

cdef class Sequence(object):
    cdef:
        vector[Example] examples

cdef double pred_cost(Example example)