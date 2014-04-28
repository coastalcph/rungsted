cdef class Dataset(object):
    cdef:
        char *quadratic
        char *ignore
        int nnz
        int hash_bits
        int n_labels


cdef class DataBlock(object):
    cdef:
        int[::1] index
        double[::1] val
        public double[::1] cost
        int size
        int next_i
        int example_start
        int is_full
        int n_labels

    cdef void start_example(self)
    cdef DataBlock copy_rest_to_new(self)


cdef class Example(object):
    cdef:
        char *id
        DataBlock block
        public int[:] index
        public double[:] val
        int length
        double importance
        public double[:] cost
        public Example next
        public Example prev
        int offset

    cdef void move_to_new_block(self)
    cdef init_views(self)
    cpdef int flat_label(self)
