from input cimport Example
from feat_map cimport FeatMap

cdef double e_score(Example *example, int label, FeatMap feat_map, double[::1] weights) nogil