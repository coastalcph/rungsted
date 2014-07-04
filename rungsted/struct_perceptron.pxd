cdef class Weights:
    cpdef public double [:, ::1] t
    cpdef public double [:, ::1] t_acc
    cpdef public int [:, ::1] t_last_update

    cpdef public double [::1] e
    cpdef public double [::1] e_acc
    cpdef public int [::1] e_last_update

    cpdef public int n_updates

    cpdef update_e(Weights self, int feat_i, double val)
    cpdef update_t(Weights self, int label_i, int label_j, double val)
