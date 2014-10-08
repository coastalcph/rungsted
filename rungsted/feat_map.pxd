from libcpp.string cimport string
from libc.stdint cimport uint32_t, int32_t
from libcpp.unordered_map cimport unordered_map

cdef uint32_t hash_str(string to_hash, int bits)

cdef class FeatMap(object):
    cdef:
        int frozen
        int next_i

    cdef int32_t feat_i(self, string)
    cdef int32_t feat_i_for_label(self, uint32_t feat_i, uint32_t label) nogil
    cpdef int32_t n_feats(self)
    cpdef int freeze(self)
    cpdef int unfreeze(self)

cdef class HashingFeatMap(FeatMap):
    cdef:
        int b
        uint32_t mask



cdef class DictFeatMap(FeatMap):
    cdef:
        public int n_labels
        object feat2index

cdef class CDictFeatMap(FeatMap):
    cdef:
        public int n_labels
        unordered_map[string, int] feat2index
