from libcpp.vector cimport vector
from libcpp.string cimport string

cdef struct dataset_s:
    vector[string] quadratic
    int * ignore
    int nnz

ctypedef dataset_s Dataset

cdef struct s_feature:
    int index
    double value

ctypedef s_feature Feature

cdef struct label_cost_s:
    int label
    double cost

ctypedef label_cost_s LabelCost

cdef struct example_s:
    Dataset dataset
    char * id_
    double importance
    vector[LabelCost] labels
    vector[Feature] features
    vector[int] constraints
    int pred_label
    int gold_label
    double pred_cost

ctypedef example_s Example

cdef double example_cost(Example example, int label)

cdef class Sequence(object):
    cdef:
        vector[Example] examples