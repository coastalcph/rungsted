from libc.stdint cimport uint32_t, int32_t, int64_t, uint64_t

cpdef uint32_t hash_feat(char* key, int b)
cpdef uint32_t hash_ints(int32_t int1, int32_t int2, int b)