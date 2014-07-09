#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

from libc.stdio cimport FILE, sscanf
from libc.stdlib cimport free, malloc

import numpy as np
cimport numpy as cnp
from cpython cimport array

from feat_map cimport FeatMap

cnp.import_array()

cdef extern from "stdio.h":
    FILE *fopen(const char *, const char *)
    int fclose(FILE *)
    ssize_t getline(char **, size_t *, FILE *)

cdef extern from "stdlib.h":
     double strtod(const char *restrict, char **restrict)

cdef extern from "string.h":
    char * strchr ( char *, int )
    char * strsep(char **, char *)
    char * strdup(const char *)
    int snprintf(char *, size_t, char *, ...)
    char * strndup(const char *, size_t)
    int isspace(int c)

cdef extern from "ctype.h":
    int isdigit(int)

DEF MAX_LEN = 2048
DEF MAX_FEAT_NAME_LEN = 1024
cdef char* DEFAULT_NS = ""


cdef class Sequence(object):
    def __cinit(self):
        pass

    def __len__(self):
        return self.examples.size()

    def __repr__(self):
        cdef Example e
        cdef int n_features = sum([e.features.size() for e in self.examples])
        cdef int n_constraints = sum([e.constraints.size() for e in self.examples])


        if self.examples.size() > 0:
            return "<Sequence ({} to {}) with {} tokens totalling " \
                   "{} features and {} constraints.>".format(self.examples.front().id_,
                                                             self.examples.back().id_,
                                                             self.examples.size(),
                                                             n_features, n_constraints)
        else:
            return "<Sequence empty>"


    property gold_labels:
        def __get__(self):
            cdef Example e
            return [e.gold_label for e in self.examples]


    property pred_labels:
        def __get__(self):
            cdef Example e
            return [e.pred_label for e in self.examples]

    property ids:
        def __get__(self):
            cdef Example e
            return [e.id_ for e in self.examples]






    def __dealloc__(self):
        cdef Example e
        for e in self.examples:
            example_free(&e)


cdef Dataset dataset_new(list quadratic_list, list ignore_list):
    cdef:
        vector[string] quadratic
        int * ignore
        int i = 0

    ignore = <int*> malloc(sizeof(int) * 256)
    for i in range(256): ignore[i] = 0

    # Initialize ignore
    for ns in ignore_list:
        if not isinstance(ns, str) or len(ns) != 1:
            raise ValueError("Invalid namespace to ignore: {}. Use one-character prefix of the namespace".format(ns))
        else:
            ignore[ord(ns)] = 1

    # Initialize quadratic
    for combo in quadratic_list:
        if not isinstance(combo, str) or len(combo) != 2:
            raise ValueError("Invalid quadratic combination: {}".format(combo))
        quadratic.push_back(combo)

    cdef Dataset dataset
    dataset = Dataset(
        quadratic=quadratic,
        ignore=ignore,
        nnz=0,
    )

    return dataset


cdef Example example_new(Dataset dataset):
    cdef vector[Feature] features
    cdef vector[int] constraints
    cdef vector[LabelCost] labels

    cdef Example example = Example(
        dataset=dataset,
        id_=NULL,
        importance=1,
        pred_label=-1,
        gold_label=-1,
        labels=labels,
        features=features,
        constraints=constraints,
        pred_cost=1.0
    )

    return example


cdef void example_free(Example * example):
    if example.id_ != NULL :
        free(example.id_)


cdef double example_cost(Example example, int label):
    cdef LabelCost label_cost
    for label_cost in example.labels:
        if label_cost.label == label:
            return label_cost.cost
    return 1.0

cdef inline void add_feature(Example * example, int index, double val):
    cdef Feature feat = Feature(index, val, 1)
    example.features.push_back(feat)


cdef LabelCost map_label(char *label_def, dict label_map):
    cdef:
        char *label_name
        char label_name_fixed[1000]
        int label = -1
        double cost = 0.0

    if strchr(label_def, ':'):
        read = sscanf(label_def, "%s:%lf", &label_name_fixed, &cost)

        if read != 2:
            raise ValueError("Invalid label: {}".format(label))
        label_name = label_name_fixed
    else:
        label_name = label_def

    # Look-up the label and allocate a new one if not found
    label = label_map.get(label_name, -1)
    if label == -1:
        label = len(label_map)
        label_map[label_def] = label

    return LabelCost(label=label, cost=cost)


cdef int parse_header(char* header, dict label_map, Example * e, int audit) except -1:
    cdef:
        char* header_elem = strsep(&header, " ")
        LabelCost label_cost

    while header_elem != NULL:
        first_char = header_elem[0]
        if first_char == '?':
            label_cost = map_label(header_elem + 1, label_map)
            # FIXME error checking
            e.constraints.push_back(label_cost.label)

        elif first_char == '\'':
            e.id_ = strdup(&header_elem[1])
        elif isdigit(first_char):
            # Tokens starting with a digit can be either
            #  - a label with optional cost, e.g. 3 and 3:0.4
            #  - an importance weight, e.g. 0.75
            # Thus if the string contains a colon, it is a label, and
            # if it contains a dot but not colon, it is an importance weight.
            if strchr(header_elem, '.') != NULL and strchr(header_elem, ':') == NULL:
                read = sscanf(header_elem, "%lf", &e.importance)
                if read != 1:
                    raise ValueError("Invalid importance weight: {}".format(header_elem))
                if audit:
                    print "imp={}".format(e.importance),
            else:
                label_cost = map_label(header_elem, label_map)
                e.labels.push_back(label_cost)
        else:
            label_cost = map_label(header_elem, label_map)
            e.labels.push_back(label_cost)


        header_elem = strsep(&header, " ")

    if len(e.labels) == 0:
        raise ValueError("No label for example")
    else:
        e.gold_label = e.labels[0].label

    if audit:
        for constraint in e.constraints:
            print "?{}".format(constraint),
        for label_cost in e.labels:
            print "{}:{}".format(label_cost.label, label_cost.cost),
        print "imp={}".format(e.importance),
        print "'{}|".format(e.id_),

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


cdef int quadratic_combinations(Example * e, int[] ns_begin, char[] ns,
                                char** feature_begin, FeatMap feat_map) except -1:
    cdef:
        int arg_i, feat_i = -1
        char arg1, arg2
        int arg1_begin, arg2_begin
        int arg1_i, arg2_i
        char combined_name[MAX_FEAT_NAME_LEN]
        int n_features = e.features.size()

    for combo in e.dataset.quadratic:
        arg1 = combo[0]
        arg2 = combo[1]

        arg1_begin = 0 if arg1 == ':' else ns_begin[<int> arg1]
        arg2_begin = 0 if arg2 == ':' else ns_begin[<int> arg2]

        if arg1_begin == -1 or arg2_begin == -1:
            continue

        # To avoid redundant feature combinations (combining
        # a feature with itself or creating both xy and yx for
        # two features x and y), we require the index of the feature
        # in the namespace given by arg2 to be strictly larger than the
        # index of the arg1 feature. That is: arg2_i > arg1_i.
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

                feat_i = feat_map.feat_i(combined_name)
                if feat_i >= 0:
                    add_feature(e, feat_i, e.features[arg1_i].value * e.features[arg2_i].value)

    return 0


cdef int parse_features(char* feature_str, Example * e, FeatMap feat_map, int audit) except -1:
    cdef:
        char ns[MAX_LEN]
        char * feature_begin[MAX_LEN]

        # Indexes of the beginning of the namespace
        int ns_begin[255] # Maximum value of char

        int feat_i = -1

        int n_features = 0
        double cur_ns_mult = 1.0
        double cur_val = 1.0

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
                raise ValueError("Number of features on line exceeds maximum allowed (defined by MAX_LEN)")

            if not e.dataset.ignore[<int> cur_ns_first]:
                # Trim space
                while isspace(feat_and_val[0]):
                    feat_and_val += 1

                if feat_and_val[0] != "\0":
                    cur_val = separate_and_parse_val(feat_and_val) * cur_ns_mult
                    snprintf(ns_and_feature_name, MAX_FEAT_NAME_LEN, "%s^%s", cur_ns, feat_and_val)
                    feat_i = feat_map.feat_i(ns_and_feature_name)

                    if audit:
                        print "{}=>{}:{}".format(ns_and_feature_name, feat_i, cur_val),

                    if feat_i >= 0:
                        add_feature(e, feat_i, cur_val)

                        feature_begin[n_features] = feat_and_val
                        ns[n_features] = cur_ns_first
                        n_features += 1

        feat_and_val = strsep(&feature_str, " ")

    quadratic_combinations(e, ns_begin, ns, feature_begin, feat_map)

    return 0

def read_vw_seq(filename, FeatMap feat_map, quadratic=[], ignore=[], labels=None, audit=False):
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
        char *header
        Dataset dataset
        Sequence seq

    label_map = {}
    if labels:
        for i, label_name in enumerate(labels):
            label_map[label_name] = i

    dataset = dataset_new(quadratic, ignore)
    seqs = []

    e = example_new(dataset)
    seq = Sequence()

    cfile = fopen(filename, "rb")
    if cfile == NULL:
        raise ValueError(2, "No such file or directory: '%s'" % filename)

    while True:
        read = getline(&line, &l, cfile)
        if read == -1:
            if seq.examples.size() > 0:
                seqs.append(seq)
            break

        # Replace newline
        if line[read-1] == '\n':
            line[read-1] = '\0'

        if len(line) >= 1:
            e = example_new(dataset)

            bar_pos = strchr(line, '|')
            if bar_pos == NULL:
                raise ValueError("Missing | character in example")

            header = strndup(line, bar_pos - line)
            parse_header(header, label_map, &e, audit)
            free(header)

            parse_features(bar_pos, &e, feat_map, audit)

            # Add constant feature
            add_feature(&e, feat_map.feat_i("^Constant"), 1)

            seq.examples.push_back(e)

            if audit:
                print ""

        else:
            # Empty line. Multiple empty lines after each other are ignored
            if seq.examples.size() > 0:
                seqs.append(seq)
                seq = Sequence()

    free(line)

    # Create label list from label map
    rev_map = dict((v, k) for k, v in label_map.items())
    labels = [rev_map[i] for i in range(len(rev_map))]

    return seqs, labels
