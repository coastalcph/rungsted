#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

from libc.stdint cimport uint32_t, int32_t, int64_t, uint64_t

cdef extern from "string.h":
    char * strncpy(char *, char *, size_t)
    int strlen(char *)
    void * memset(void *, int, size_t)

cdef extern from "MurmurHash3.h":
    void MurmurHash3_x86_32  (void *, int, uint32_t, void *)


DEF MURMUR_SEED = 100
# MAX_PADDED_LEN Should be a multiple of 4
DEF MAX_PADDED_LEN = 4*512


cpdef inline uint32_t hash_feat(char* key, int b):
    cdef uint32_t out = 0
    cdef int pad_len = 0
    cdef char padded_key[MAX_PADDED_LEN]
    cdef int padded_len
    cdef uint32_t mask = ((1 << b) - 1)
    cdef int key_len

    key_len = len(key)
    # Truncate key
    if key_len > MAX_PADDED_LEN:
        key_len = MAX_PADDED_LEN

    # Pad the string with the null byte making the length a multiple of 4.
    # padded_len never exceeds MAX_PADDED_LEN, because the constant is a multiple of 4
    padded_len = key_len + (key_len % 4)
    memset(padded_key, 0, padded_len)

    # Write the string on top of the padding
    strncpy(padded_key, key, key_len)

    MurmurHash3_x86_32(padded_key, padded_len, MURMUR_SEED, &out)

    return out & mask

cpdef inline uint32_t hash_ints(int32_t int1, int32_t int2, int b):
    cdef uint32_t out = 0
    cdef uint32_t mask = ((1 << b) - 1)
    cdef uint64_t input

    # Combine the bits of the two 32-bit integers into a 64-bit int
    input = int1
    input <<= 32
    input |= int2

    MurmurHash3_x86_32(&input, sizeof(uint64_t), MURMUR_SEED, &out)

    return out & mask

# import numpy as np
# arr1 = np.random.random_integers(0, 2**18, size=n_pairs).astype(np.int32)
# arr2 = np.random.random_integers(0, 2**18, size=n_pairs).astype(np.int32)
# In [5]: %timeit perf_hash_ints(arr1, arr2, 1000, 1000)
# 100 loops, best of 3: 12.3 ms per loop
def perf_hash_ints(int[:] arr1, int[:] arr2, n_pairs, n_rounds):
    cdef int i, j

    for i in range(n_rounds):
        for j in range(n_pairs):
            hash_ints(arr1[j], arr2[j], 18)

