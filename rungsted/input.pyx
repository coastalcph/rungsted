

from libc.stdio cimport FILE, sscanf
from libc.stdint cimport uint8_t
from libc.stdlib cimport free, malloc
from libcpp.string cimport string
from libcpp cimport bool


from cython.operator cimport dereference as deref
import numpy as np
cimport numpy as cnp
from cpython cimport array

from feat_map cimport FeatMap, hash_str

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

from libc.stdlib cimport rand


DEF MAX_LEN = 2048
DEF MAX_FEAT_NAME_LEN = 1024
cdef char* DEFAULT_NS = ""

cdef FeatMap feature_map_global


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


cpdef uint8_t[::1] count_group_sizes(list sequences):
    cdef:
        Sequence seq
        Example example
        Feature feat
        int max_group = -1
        set group_feature_pairs = set()

    for seq in sequences:
        for example in seq.examples:
            for feat in example.features:
                if feat.group > max_group:
                    max_group = feat.group
                group_feature_pairs.add((feat.group, feat.index))

    cdef uint8_t [::1] out = np.zeros(max_group + 1, dtype=np.uint8)
    cdef int group, index
    for group, index in group_feature_pairs:
        out[group] += 1

    return out

cpdef dropout_groups(list sequences, uint8_t[::1] group_sizes):
    cdef:
        int total_features = sum(group_sizes)
        uint8_t [::1] blocked_groups = np.zeros_like(group_sizes)

        long [::1] visit_order
        int block_count
        int currently_blocked = 0

    # Block 10 % of features (should be adjustable)
    block_count = int(total_features * 0.1)

    cdef int i
    while currently_blocked <= block_count:
        i = rand() % group_sizes.shape[0]
        # Already blocked?
        if blocked_groups[i] == 1:
            continue
        if group_sizes[i] > 0:
            blocked_groups[i] = 1
            currently_blocked += group_sizes[i]

    block_groups(sequences, blocked_groups)

cpdef int block_groups(list sequences, uint8_t[::1] blocked_groups):
    cdef:
        Sequence seq
        Example *example
        Feature *feat
        int n_blocked = 0

    cdef int i, j
    for seq in sequences:
        for i in range(seq.examples.size()):
            example = &(seq.examples[i])
            for j in range(example.features.size()):
                feat = &(seq.examples[i].features[j])
                if blocked_groups[feat.group] == 1:
                    n_blocked += 1
                    feat.active = 0
                else:
                    feat.active = 1
    return n_blocked

cdef Dataset dataset_new(list quadratic_list, list ignore_list):
    cdef:
        vector[string] quadratic
        int * ignore
        int i = 0
        uint8_t [:, ::1] combos = np.zeros((256, 256), dtype=np.uint8)

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
        combos[<int> combo[0], <int> combo[1]] = 1
        combos[<int> combo[1], <int> combo[0]] = 1


    cdef Dataset dataset
    dataset = Dataset(
        quadratic=quadratic,
        ignore=ignore,
        nnz=0,
        combos=combos
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

cdef LabelCost map_label(string label_def, dict label_map):
    cdef:
        string label_name
        int label = -1
        double cost = 0.0
        int colon_pos

    colon_pos = label_def.find(':')
    if colon_pos != -1:
        label_name = label_def.substr(0, colon_pos)
        cost = strtod(label_def.substr(colon_pos).c_str(), NULL)
    else:
        label_name = label_def

    # Look-up the label and allocate a new one if not found
    label = label_map.get(label_name, -1)
    if label == -1:
        label = len(label_map)
        label_map[label_def] = label

    return LabelCost(label=label, cost=cost)


cdef vector[string] tokenize_header(string header):
    cdef:
        vector[string] results
        int begin = 0
        int found
        string space = " "

    while True:
        found = header.find(space, begin+1)
        if found == -1:
            results.push_back(header.substr(begin, header.size() - begin))
            break

        results.push_back(header.substr(begin, found - begin))
        begin = found + 1

    return results


cdef int parse_header(string header, dict label_map, Example * e, int audit) except -1:
    cdef:
        int i = 0
        int start = 0
        int next_break = -1
        vector[string] tokens = tokenize_header(header)

    for token in tokens:
        if token[0] == '?':
            label_cost = map_label(token.substr(1), label_map)
            e.constraints.push_back(label_cost.label)
        elif token[0] == '\'':
            e.id_ = strdup(token.substr(1).c_str())
        elif isdigit(token[0]):
            # Tokens starting with a digit can be either
            #  - a label with optional cost, e.g. 3 and 3:0.4
            #  - an importance weight, e.g. 0.75
            # Thus if the string contains a colon, it is a label, and
            # if it contains a dot but not colon, it is an importance weight.
            if token.find(".") and token.find(":") != -1:
                e.importance = strtod(token.c_str(), NULL)

                if e.importance < 0:
                    raise ValueError("Invalid importance weight: {}".format(e.importance))
            else:
                label_cost = map_label(token, label_map)
                e.labels.push_back(label_cost)
        else:
            label_cost = map_label(token, label_map)
            e.labels.push_back(label_cost)

    if len(e.labels) == 1:
        e.gold_label = e.labels[0].label


    if audit:
        for constraint in e.constraints:
            print "?{}".format(constraint),
        for label_cost in e.labels:
            print "{}:{}".format(label_cost.label, label_cost.cost),
        print "imp={}".format(e.importance),
        print "'{}|".format(e.id_),

    return 0


cdef struct PartialExample:
    Example *example
    string ns_name
    string feat_name
    string feat_group
    double feat_val
    double ns_mult

    vector[string] names


cdef void add_feature(Example *example, int feat_i, double value, int active, int group):
    cdef Feature feat
    feat.index = feat_i
    feat.value = value
    feat.active = active
    feat.group = group
    example.features.push_back(feat)

cdef void add_partial(Example *example, PartialExample *partial, int audit):
    global feature_map_global
    cdef FeatMap feat_map = feature_map_global
    assert feature_map_global is not None

    # cdef string full_feat_name

    if not partial.ns_name.empty() and example.dataset.ignore[<int> partial.ns_name[0]]:
        partial.feat_group.clear()
        return

    partial.feat_name.push_back("^")
    partial.feat_name += partial.ns_name
    # full_feat_name = partial.ns_name
    # full_feat_name.push_back("^")
    # full_feat_name += partial.feat_name
    feat_i = feat_map.feat_i(partial.feat_name.c_str())

    cdef Feature feat
    if feat_i >= 0:
        partial.names.push_back(partial.feat_name)

        feat.index = feat_i
        feat.value = partial.ns_mult * partial.feat_val
        feat.active = 1
        if not partial.feat_group.empty():
            feat.group = hash_str(partial.feat_group, 22)
        else:
            feat.group = hash_str(partial.feat_name, 22)

        example.features.push_back(feat)

        if audit:
            print "{}:{}@{}=>{}".format(partial.feat_name, feat.value, feat.group, feat.index),


    partial.feat_group.clear()



# cdef int parse_features(string feature_str, Example * e, FeatMap feat_map, int audit, PartialExample *partial) except -1:
#     cdef:
#         # Indexes of the beginning of the namespace
#         int ns_begin[255] # Maximum value of char
#         double cur_ns_mult = 1
#         string cur_ns
#         string cur_feat_name
#         string cur_feat_group
#         double cur_feat_val = 1.0
#
#         # States
#         short NS_NAME = 1
#         short FEATURE_NAME = 2
#         short FEATURE_TRAILER = 4
#         short NS_VAL = 8
#         short OUTSIDE = 16
#         short FEATURE_VAL = 32
#         short FEATURE_GROUP = 64
#         short INVALID = 128
#
#         # Types of tokens
#         short SPACE = 1
#         short BAR = 2
#         short COLON = 4
#         short DIGIT = 8
#         short AT = 16
#         short OTHER = 32
#         short FINAL = 64
#
#         int i
#
#
#     partial.feat_val = 1.0
#     partial.ns_mult = 1.0
#     # Initialize with -1, a value indicating the namespace is not present
#     # in the current example
#     for i in range(256): ns_begin[i] = -1
#
#     cdef string buf = string()
#
#     cdef char c
#     cdef int token
#     cdef int state = OUTSIDE
#     for i in range(feature_str.size()):
#         c = feature_str[i]
#         if isspace(c):
#             token = SPACE
#         elif c == '|':
#             token = BAR
#         elif c == ':':
#             token = COLON
#         elif isdigit(c):
#             token = DIGIT
#         elif c == '@':
#             token = AT
#         else:
#             token = OTHER
#
#         if state == FEATURE_NAME:
#             if token & (COLON|SPACE|FINAL):
#                 partial.feat_name = buf
#                 partial.feat_val = 1.0
#                 buf.clear()
#                 if token == COLON:
#                     state = FEATURE_TRAILER
#                 else:
#                     state = OUTSIDE
#                     add_partial(e, partial, feat_map, audit)
#             else:
#                 buf += c
#
#         elif state == FEATURE_VAL:
#             if token & (SPACE|FINAL|AT):
#                 partial.feat_val = strtod(buf.c_str(), NULL)
#                 buf.clear()
#
#                 if token & (SPACE|FINAL):
#                     add_partial(e, partial, feat_map, audit)
#                     state = OUTSIDE
#                 else:
#                     state = FEATURE_GROUP
#
#             elif token == DIGIT or c == '.':
#                 buf += c
#             else:
#                 state = INVALID
#
#         elif state == FEATURE_TRAILER:
#             if token == AT:
#                 state = FEATURE_GROUP
#             elif token == DIGIT:
#                 state = FEATURE_VAL
#                 buf += c
#             else:
#                 state = INVALID
#
#         elif state == FEATURE_GROUP:
#             if token & (SPACE|FINAL):
#                 partial.feat_group = buf
#                 buf.clear()
#                 add_partial(e, partial, feat_map, audit)
#                 state = OUTSIDE
#             else:
#                 buf.push_back(c)
#
#         elif state == NS_NAME:
#             if token & (SPACE|COLON):
#                 partial.ns_name = buf
#                 partial.ns_mult = 1.0
#                 buf.clear()
#                 state = OUTSIDE if token == SPACE else NS_VAL
#             else:
#                 buf += c
#
#         elif state == NS_VAL:
#             if token & SPACE:
#                 partial.ns_mult = strtod(buf.c_str(), NULL)
#                 buf.clear()
#                 state = OUTSIDE
#             elif token == DIGIT or c == '.':
#                 buf += c
#             else:
#                 state = INVALID
#
#         elif state == OUTSIDE:
#             if token == BAR:
#                 state = NS_NAME
#             elif token != SPACE:
#                 buf += c
#                 state = FEATURE_NAME
#
#         else:
#             state = INVALID
#
#         if state == INVALID:
#             raise ValueError("Malformed input: {}>>>>{}<<<<{} ".format(
#                 feature_str.substr(0, i), chr(feature_str[i]), feature_str.substr(i+1, feature_str.length())))
#
#
#     return 0



cdef int parse_features2(string feature_str, Example * e, int audit, PartialExample *partial) except -1:
    cdef:
        short inside_feature = 0
        int i = 0, found, head = 0, read
        char c, next_c
        double parsed_val
        string number_like = "0123456789.+-"
        string space = " \n"
        string space_or_colon = ": \n"

    # print "\ngot", feature_str

    partial.feat_val = 1.0
    partial.ns_mult = 1.0
    # Initialize with -1, a value indicating the namespace is not present
    # in the current example
    while i < feature_str.size():
        c = feature_str[i]

        # We're about to read one of
        #   * a feature value (number)
        #   * a group name (prefixed by @)
        if inside_feature:
            # Check if next character is digit-like
            if feature_str.find_first_of(number_like, i) == i:
                parsed_val = strtod(feature_str.c_str() + i, NULL)
                if not parsed_val:
                    raise ValueError("Feature value is not a number: '{}'".format(feature_str.substr(i)))
                else:
                    partial.feat_val = parsed_val
                    found = feature_str.find_first_of(space, i)
                    assert found != -1
                    if feature_str[found] == ' ':
                        inside_feature = 0
                        add_partial(e, partial, audit)
                    else:
                        inside_feature = 1

                    i = found

            elif c == '@':
                found = feature_str.find_first_of(space, i + 1)
                assert found != -1
                partial.feat_group = feature_str.substr(i + 1, found - i - 1)
                # print "group '{}'".format(partial.feat_group)
                inside_feature = 0
                add_partial(e, partial, audit)
                i = found
            else:
                raise ValueError("Invalid feature declaration (after ':')")

        elif c == '|':
            found = feature_str.find_first_of(space_or_colon, i)
            assert found != -1
            partial.ns_name = feature_str.substr(i + 1, found - i - 1)
            # print "found ns '{}'".format(partial.ns_name)
            i = found

            if feature_str[i] == ':':
                found = feature_str.find_first_of(space, i)
                assert found != -1
                parsed_val = strtod(feature_str.c_str() + i, NULL)
                if not parsed_val:
                    raise ValueError("Invalid namespace multiplier value")
                partial.ns_mult = parsed_val
                # print "ns multiplier", partial.ns_mult
                i = found
        elif c == ' ':
            # Skip space
            i += 1
        else:
            # Feature
            found = feature_str.find_first_of(space_or_colon, i)
            assert found != -1
            partial.feat_name = feature_str.substr(i, found - i)
            if feature_str[found] == ':' and found < feature_str.size():
                inside_feature = 1
            else:
                add_partial(e, partial, audit)

            i = found + 1

    return 0


def read_vw_seq(filename, FeatMap feat_map, quadratic=[], ignore=[], labels=None, audit=False, require_labels=False):
    cdef:
        char* fname
        FILE* cfile
        char * line = NULL
        unsigned int bar_pos
        size_t l = 0
        ssize_t read
        double instance_weight
        size_t id_len
        Example e
        string header
        const char *c_header
        Dataset dataset
        Sequence seq
        PartialExample partial
        string line_str
        string feature_section
        string new_line = "\n"

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

    global feature_map_global
    feature_map_global = feat_map
    while True:
        read = getline(&line, &l, cfile)
        if read == -1:
            if seq.examples.size() > 0:
                seqs.append(seq)
            break

        line_str = string(line)
        if line_str[line_str.size() - 1] != "\n":
            line_str.push_back("\n")

        if line_str.size() > 1:
            e = example_new(dataset)

            bar_pos = line_str.find('|')

            if bar_pos == -1:
                raise ValueError("Missing | character in example")

            header = line_str.substr(0, bar_pos)
            parse_header(header, label_map, &e, audit)
            if require_labels and e.labels.size() == 0:
                raise ValueError("Missing label in example: {}".format(line))


            feature_section = line_str.substr(bar_pos + 1)

            parse_features2(feature_section, &e, audit, &partial)

            # Add constant feature
            add_feature(&e, feat_map.feat_i("^Constant"), 1, 1, -1)

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
