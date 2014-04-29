from libc.stdint cimport uint32_t, int32_t

cdef class FeatMap(object):
    cdef:
        int frozen

    cdef int32_t feat_i(self, char * feat)
    cdef int32_t feat_i_for_label(self, uint32_t feat_i, uint32_t label)
    cpdef int32_t n_feats(self)
    cpdef int freeze(self)
    cpdef int unfreeze(self)

cdef class HashingFeatMap(FeatMap):
    cdef:
        int b
        uint32_t mask



cdef class DictFeatMap(FeatMap):
    cdef:
        int n_labels
        int next_i
        object feat2index



