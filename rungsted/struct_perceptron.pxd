from input cimport Example
from feat_map cimport FeatMap
from input cimport Sequence
from weights cimport WeightVector

cdef double e_score(Example *example, int label, FeatMap feat_map, double[::1] weights) nogil
