from rungsted.input cimport Example
from rungsted.input cimport Sequence
from rungsted.feat_map cimport FeatMap
from rungsted.weights cimport WeightVector

cdef double e_score(Example *example, int label, FeatMap feat_map, double[::1] weights) nogil
