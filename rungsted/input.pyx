from libc.stdio cimport FILE, sscanf
from libc.stdint cimport uint32_t, int64_t
from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as cnp
from numpy cimport PyArray_SimpleNewFromData
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cython cimport view
from cpython cimport PyObject, Py_INCREF, Py_DECREF
from cpython cimport array

cimport cython
import sys

import hashing
from hashing cimport hash_feat

cnp.import_array()

cdef extern from "stdio.h":
    FILE *fopen(const char *, const char *)
    int fclose(FILE *)
    ssize_t getline(char **, size_t *, FILE *)

cdef extern from "stdlib.h":
     double strtod(const char *restrict, char **restrict)

cdef extern from "string.h":
    char * strchr ( char *, int )
    char * strtok(char *, char *)
    char * strsep(char **, char *)
    char * strdup(const char *)
    char * strcpy(char *, char *)
    char * strncpy(char *, char *, size_t)
    int strlen(char *)
    int snprintf(char *, size_t, char *, ...)
    void * memset(void *, int, size_t)
    void * memcpy(void *, const void *, size_t)
    int asprintf(char **, char *, ...)
    size_t strlcpy(char *, const char *, size_t)
    char * strndup(const char *, size_t)

cdef extern from "ctype.h":
    int isdigit(int)

DEF MAX_LEN = 2048
DEF MAX_FEAT_NAME_LEN = 1024
cdef char* DEFAULT_NS = ""
DEF HASH_BITS = 18
DEF BLOCK_SIZE = 50*1000
# DEF BLOCK_SIZE = 50

cdef class DataBlock(object):
    def __cinit__(self, int size, int n_labels):

        self.size = size
        # Initializing the arrays with array.clone(...) is fast and avoids memory copy.
        # However, it raises a deprecated error, which is currently ignored in the build
        # by the -Wno-deprecated-writable-strings flag
        self.index = array.clone(array.array("i"), size, False)
        self.val = array.clone(array.array("d"), size, False)
        self.next_i = 0
        self.example_start = 0
        self.n_labels = n_labels

    cdef void start_example(self):
        self.example_start = self.next_i

    cdef DataBlock copy_rest_to_new(self):
        cdef DataBlock new_block = DataBlock(self.size, self.n_labels)
        cdef int old_i, new_i, old_size, j

        # Check if remaining data fits in new DataBlock
        old_size = self.next_i - self.example_start
        assert old_size <= self.size

        for old_i in range(self.example_start, self.next_i):
            new_i = old_i - self.example_start
            new_block.index[new_i] = self.index[old_i]
            new_block.val[new_i] = self.val[old_i]

        new_block.example_start = 0
        new_block.next_i = self.next_i - self.example_start
        new_block.is_full = new_block.next_i == new_block.size

        return new_block

# Although this function logically belongs to the DataBlock class, we define
# it here, because methods attached to classes cannot be inlined
cdef inline int add_feature(Example e, int index, double val) except -1:
    if e.block.is_full:
        e.move_to_new_block()
        if e.block.is_full:
            raise StandardError("Number of features on line exceeds block size")

    e.block.index[e.block.next_i] = index
    e.block.val[e.block.next_i] = val
    e.block.next_i = e.block.next_i + 1
    e.block.is_full = e.block.next_i == e.block.size

    return 0

cdef class Dataset(object):
    def __init__(self, n_labels, hash_bits=18, quadratic=[], ignore=[]):
        for combo in quadratic:
            if not isinstance(combo, str) or len(combo) != 2:
                raise StandardError("Invalid quadratic combination: {}".format(combo))
        quadratic_str = "".join(quadratic)
        self.quadratic = quadratic_str

        for ns in ignore:
            if not isinstance(ns, str) or len(ns) != 1:
                raise StandardError("Invalid namespace to ignore: {}. Use one-character prefix of the namespace".format(ns))
        ignore_str = "".join(ignore)
        self.ignore = ignore_str
        self.nnz = 0
        self.n_labels = n_labels
        self.hash_bits = hash_bits


cdef class Example(object):
    def __init__(self, DataBlock block, int offset):
        self.block = block
        self.offset = offset

        # Initialize cost array with 1.0
        self.cost = array.clone(array.array("d"), block.n_labels, False)
        cdef int i
        for i in range(block.n_labels):
            self.cost[i] = 1.0


    cdef void move_to_new_block(self):
        cdef DataBlock new_block

        new_block = self.block.copy_rest_to_new()
        self.block = new_block

    cdef init_views(self):
        self.index = self.block.index[self.offset:self.offset+self.length]
        self.val = self.block.val[self.offset:self.offset+self.length]


    cpdef int flat_label(self):
        cdef int i
        for i in range(self.block.n_labels):
            if self.cost[i] == 0:
                return i + 1


    def __dealloc__(self):
        if self.id_: free(self.id_)

    def __repr__(self):
        return "<Example id={} with " \
               "{} features.>".format(self.id_, self.length)


cdef int parse_header(char* header, Example e) except -1:
    cdef:
        int label
        char* header_elem = strsep(&header, " ")
        double cost, importance

    while header_elem != NULL:
        first_char = header_elem[0]
        if first_char == '?':
            read = sscanf(header_elem + 1, "%i", &label)
            if read == 1:
                pass
                #constraints.append(label)
        elif first_char == '\'':
            e.id_ = strdup(&header_elem[1])
        elif isdigit(first_char):
            # Tokens starting with a digit can be either
            #  - a label with optional cost, e.g. 3 and 3:0.4
            #  - an importance weight, e.g. 0.75
            # Thus if the string contains a colon, it is a label, and
            # if it contains a dot but not colon, it is an importance weight.
            if strchr(header_elem, ':') != NULL:
                read = sscanf(header_elem, "%i:%lf", &label, &cost)
                if read == 2:
                    if 0 < label <= e.block.n_labels:
                        e.cost[label-1] = cost
                    else:
                        raise StandardError("Invalid label: {}".format(label))
                else:
                    raise StandardError("Invalid label specification: {}".format(header_elem))
            elif strchr(header_elem, '.') != NULL:
                read = sscanf(header_elem, "%lf", &e.importance)
                if read != 1:
                    raise StandardError("Invalid importance weight: {}".format(header_elem))
            else:
                read = sscanf(header_elem, "%i", &label)
                if read == 1:
                    if 0 < label <= e.block.n_labels:
                        e.cost[label-1] = 0.0
                    else:
                        raise StandardError("Invalid label: {}".format(label))
                else:
                    raise StandardError("Invalid importance weight: {}".format(header_elem))

        header_elem = strsep(&header, " ")

    return 0

cdef double separate_and_parse_val(char* string_with_value) except -1:
    """Looks for a decimal value at the end of the string.
    The decimal number should be preceeded by a colon.
    If no colon is found, a default value of 1.0 is returned.
    If there is a colon but it is not followed by a number,
    the function raises an error (TODO).

    If found, the colon is replaced with a '\0' byte.
    """
    colon_pos = strchr(string_with_value, ":")

    if colon_pos != NULL:
        colon_pos[0] = "\0"
        if colon_pos[1] == "\0":
            return -1
        else:
            # TODO check return value
            return strtod(colon_pos + 1, NULL)
    else:
        return 1


cdef int quadratic_combinations(char* quadratic, Example e, int[] ns_begin, char[] ns, int n_features,
                                char** feature_begin) except -1:
    cdef:
        int arg_i
        char arg1, arg2
        int arg1_begin, arg2_begin
        int arg1_i, arg2_i
        char combined_name[MAX_FEAT_NAME_LEN]
        int n_combos = 0

    for arg_i in range(0, len(quadratic), 2):
        arg1 = quadratic[arg_i]
        arg2 = quadratic[arg_i+1]

        arg1_begin = 0 if arg1 == ':' else ns_begin[<int> arg1]
        arg2_begin = 0 if arg2 == ':' else ns_begin[<int> arg2]

        if arg1_begin == -1 or arg2_begin == -1:
            continue

        # To avoid redundant feature combinations (combining
        # a feature with itself or creating both xy and yx for
        # two features x and y), we require the index of the feature
        # in the namespace given by arg2 to be strictly larger than the
        # index of the arg1 feature. That is: arg2_i > arg1_i.
        # The problem arises when combining a namespace with itself
        # (arg1 == arg2) or using the all-namespaces symbol
        # (arg1 == ':' or arg2 == ':').
        #
        # Therefore, we may need to swap to arguments to ensure that
        # the arg2 namespace is after arg1
        if arg1_begin > arg2_begin:
            arg1, arg2 = arg2, arg1
            arg1_begin, arg2_begin = arg2_begin, arg1_begin

        for arg1_i in range(arg1_begin, n_features):
            if arg1 != ':' and ns[arg1_i] != arg1:
                break
            for arg2_i in range(max(arg1_i+1, arg2_begin), n_features):
                if arg2 != ':' and ns[arg2_i] != arg2:
                    break

                snprintf(combined_name, MAX_FEAT_NAME_LEN, "%c^%s+%c^%s",
                         ns[arg1_i], feature_begin[arg1_i],
                         ns[arg2_i], feature_begin[arg2_i])


                add_feature(e,
                            hash_feat(combined_name, HASH_BITS),
                            e.val[arg1_i] * e.val[arg2_i])
                n_combos += 1

    return n_combos



cdef parse_features(char* feature_str, Example e, char* quadratic):
    cdef:
        char ns[MAX_LEN]
        char * feature_begin[MAX_LEN]

        # Indexes of the beginning of the namespace
        int ns_begin[255] # Maximum value of char

        int n_features = 0
        double cur_ns_mult = 1.0

        char * cur_ns = DEFAULT_NS
        char cur_ns_first = 0
        char * feat_and_val = NULL
        char * pair_separator = NULL
        char ns_and_feature_name[MAX_FEAT_NAME_LEN]

    # Initialize with -1, a value indicating the namespace is not present
    # in the current example
    for i in range(256): ns_begin[i] = -1

    feat_and_val = strsep(&feature_str, " ")
    while feat_and_val != NULL:
        # The feat_and_val is a namespace
        if feat_and_val[0] == '|':
            if feat_and_val[1] == "\0":
                cur_ns = DEFAULT_NS
                cur_ns_first = "_"
            else:
                cur_ns = feat_and_val + 1
                cur_ns_first = feat_and_val[1]

            ns_begin[<int> cur_ns_first] = n_features
            cur_ns_mult = separate_and_parse_val(feat_and_val)

        else:
            if n_features == MAX_LEN:
                raise StandardError("Number of features on line exceeds maximum allowed (defined by MAX_LEN)")

            snprintf(ns_and_feature_name, MAX_FEAT_NAME_LEN, "%s^%s", cur_ns, feat_and_val)
            add_feature(e,
                        hash_feat(ns_and_feature_name, HASH_BITS),
                        cur_ns_mult * separate_and_parse_val(feat_and_val))

            feature_begin[n_features] = feat_and_val
            ns[n_features] = cur_ns_first
            n_features += 1

        feat_and_val = strsep(&feature_str, " ")

    n_features += quadratic_combinations(quadratic, e, ns_begin, ns, n_features, feature_begin)

    return n_features


def read_vw_seq(filename, n_labels, quadratic=[], ignore=[]):
    cdef:
        char* fname
        FILE* cfile
        char * line = NULL
        char * bar_pos
        size_t l = 0
        ssize_t read
        double instance_weight
        size_t id_len
        Example e
        Example prev_e = None
        char *header
        Dataset dataset


    dataset = Dataset(n_labels, ignore=ignore, quadratic=quadratic)
    seqs = []
    seq = []

    e = Example(DataBlock(BLOCK_SIZE, n_labels), 0)

    cfile = fopen(filename, "rb")
    if cfile == NULL:
        raise StandardError(2, "No such file or directory: '%s'" % filename)

    while True:
        read = getline(&line, &l, cfile)
        if read == -1:
            if len(seq) > 0:
                seqs.append(seq)
            break

        # m = header_re.match(line)
        if len(line) > 1:
            e.block.start_example()
            e = Example(e.block, e.block.example_start)

            bar_pos = strchr(line, '|')
            if bar_pos == NULL:
                raise StandardError("Missing | character in example")

            header = strndup(line, bar_pos - line)
            parse_header(header, e)
            free(header)

            features_parsed = parse_features(bar_pos, e, dataset.quadratic)
            e.length = features_parsed
            e.init_views()

            seq.append(e)

        else:
            if len(seq) > 0:
                seqs.append(seq)
                seq = []

    return seqs
