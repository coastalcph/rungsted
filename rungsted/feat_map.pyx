#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

from libc.stdint cimport uint32_t, int32_t, int64_t, uint64_t

cdef extern from "string.h":
    char * strncpy(char *, char *, size_t) nogil
    int strlen(char *) nogil
    void * memset(void *, int, size_t) nogil

cdef extern from "MurmurHash3.h":
    void MurmurHash3_x86_32  (void *, int, uint32_t, void *) nogil


cdef class FeatMap(object):
    cdef int32_t feat_i(self, char * feat):
        return -1
    cdef int32_t feat_i_for_label(self, uint32_t feat_i, uint32_t label):
        return -1
    cpdef int32_t n_feats(self):
        return -1
    cpdef int freeze(self):
        self.frozen = 1
        return self.frozen
    cpdef int unfreeze(self):
        self.frozen = 0
        return self.frozen


DEF MURMUR_SEED = 100
# MAX_PADDED_LEN Should be a multiple of 4
DEF MAX_PADDED_LEN = 4*512

cdef class HashingFeatMap(FeatMap):
    def __init__(self, int b):
        self.b = b
        self.mask = ((1 << b) - 1)

    cdef int32_t feat_i(self, char * feat):
        cdef:
            uint32_t out = 0
            int pad_len = 0
            char padded_key[MAX_PADDED_LEN]
            int padded_len
            int key_len

        key_len = strlen(feat)
        # Truncate key
        if key_len > MAX_PADDED_LEN:
            key_len = MAX_PADDED_LEN

        # Pad the string with the null byte making the length a multiple of 4.
        # padded_len never exceeds MAX_PADDED_LEN, because the constant is a multiple of 4
        padded_len = key_len + (key_len % 4)
        memset(padded_key, 0, padded_len)

        # Write the string on top of the padding
        strncpy(padded_key, feat, key_len)

        MurmurHash3_x86_32(padded_key, padded_len, MURMUR_SEED, &out)

        return out & self.mask

    cdef int32_t feat_i_for_label(self, uint32_t feat_i, uint32_t label):
        cdef:
            uint32_t out = 0
            uint64_t input

        # Combine the bits of the two 32-bit integers into a 64-bit int
        input = label
        input <<= 32
        input |= feat_i

        MurmurHash3_x86_32(&input, sizeof(uint64_t), MURMUR_SEED, &out)

        return out & self.mask

    cpdef int32_t n_feats(self):
        return 2**self.b


cdef class DictFeatMap(FeatMap):
    property feat2index_:

        def __get__(self):
            return self.feat2index

        def __set__(self, value):
            self.feat2index = value


    def __init__(self, int n_labels):
        self.next_i = 0
        self.n_labels = n_labels
        self.feat2index = {}

    cdef int32_t feat_i(self, char * feat):
        cdef int32_t key
        key = self.feat2index.get(feat, -1)
        if key != -1 or self.frozen == 1:
            return key
        else:
            key = self.next_i
            self.feat2index[feat] = key
            self.next_i += 1
            return key

    cdef int32_t feat_i_for_label(self, uint32_t feat_i, uint32_t label):
        # The weight weight has `n_labels` sections, each with `next_i` entries
        return self.next_i * (label - 1) + feat_i

    cpdef int32_t n_feats(self):
        return self.next_i * self.n_labels