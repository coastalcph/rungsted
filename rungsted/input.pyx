
from libc.stdio cimport FILE, sscanf
from libc.stdint cimport uint8_t
from libc.stdlib cimport free, malloc
from libcpp.string cimport string
from libcpp cimport bool



from cython.operator cimport dereference as deref
import numpy as np
cimport numpy as cnp
from cpython cimport array

from rungsted.feat_map cimport FeatMap, hash_str

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
cdef const char * DEFAULT_ID = b"default"
# FIXME should really be imported from the string
cdef size_t npos = -1

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


    property features:
        def __get__(self):
            cdef Example e
            cdef Feature feat

            return [(feat.index, feat.value)
                    for e in self.examples
                    for feat in e.features]

    property importance_weights:
        def __get__(self):
            cdef Example e
            return [e.importance for e in self.examples]



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

    property label_costs:
        def __get__(self):
            cdef Example e
            return [[(label.label, label.cost) for label in e.labels]
                    for e in self.examples]

    def __dealloc__(self):
        cdef Example e
        for e in self.examples:
            example_free(&e)


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

    colon_pos = label_def.find(b':')
    if colon_pos != npos:
        label_name = label_def.substr(0, colon_pos)
        cost = strtod(label_def.substr(colon_pos + 1).c_str(), NULL)
    else:
        label_name = label_def

    # Look-up the label and allocate a new one if not found
    label = label_map.get(label_name, -1)
    if label == -1:
        label = len(label_map)
        label_map[label_name] = label

    return LabelCost(label=label, cost=cost)


cdef vector[string] tokenize_header(string header):
    cdef:
        vector[string] results
        int begin = 0
        int found
        string space = b' '

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
        if token[0] == b'?':
            label_cost = map_label(token.substr(1), label_map)
            e.constraints.push_back(label_cost.label)
        elif token[0] == b'\'':
            e.id_ = strdup(token.substr(1).c_str())
        elif isdigit(token[0]):
            # Tokens starting with a digit can be either
            #  - a label with optional cost, e.g. 3 and 3:0.4
            #  - an importance weight, e.g. 0.75
            # Thus if the string contains a colon, it is a label, and
            # if it contains a dot but not colon, it is an importance weight.
            if token.find(b'.') != npos and token.find(b':') == npos:
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

    # If no id was set, assign a default
    if e.id_ == NULL:
        e.id_ = strdup(DEFAULT_ID)

    if audit:
        for constraint in e.constraints:
            print("?{}".format(constraint), end=' ')
        for label_cost in e.labels:
            print("{}:{}".format(label_cost.label, label_cost.cost), end=' ')
        print("imp={}".format(e.importance), end=' ')
        print("'{}|".format(e.id_), end=' ')

    return 0


cdef struct PartialExample:
    Example *example
    string *feature_str
    string ns_name
    string feat_name
    int feat_begin
    int feat_len
    double feat_val
    double ns_mult

    vector[string] names


cdef void add_feature(Example *example, int feat_i, double value):
    cdef Feature feat
    feat.index = feat_i
    feat.value = value
    example.features.push_back(feat)

cdef void add_partial(Example *example, PartialExample *partial, int audit):
    global feature_map_global
    cdef FeatMap feat_map = feature_map_global
    assert feature_map_global is not None

    if not partial.ns_name.empty() and example.dataset.ignore[<int> partial.ns_name[0]]:
        return

    cdef int ns_name_len = partial.ns_name.size()

    partial.ns_name.push_back(b"^")
    partial.ns_name.append(deref(partial.feature_str), partial.feat_begin, partial.feat_len)
    feat_i = feat_map.feat_i(partial.ns_name)

    cdef Feature feat
    if feat_i >= 0:
        partial.names.push_back(partial.feat_name)

        feat.index = feat_i
        feat.value = partial.ns_mult * partial.feat_val

        example.features.push_back(feat)

        if audit:
            print("{}:{}=>{}".format(partial.ns_name, feat.value, feat.index), end=' ')

    partial.ns_name.resize(ns_name_len)


cdef int parse_features2(Example * e, int audit, PartialExample *partial) except -1:
    cdef:
        short inside_feature = 0
        int i = 0, found, head = 0, read
        char c
        double parsed_val
        string number_like = b"0123456789.+-"
        string space = b" \n"
        string space_or_colon = b": \n"

        const char *str_begin
        char *str_end

    partial.feat_val = 1.0
    partial.ns_mult = 1.0
    cdef string feature_str = deref(partial.feature_str)

    # print "got feature str '{}'".format(feature_str)
    # Initialize with -1, a value indicating the namespace is not present
    # in the current example
    while i < feature_str.size():
        c = feature_str[i]

        # We're about to read one of a feature value (number)
        if inside_feature:
            # Check if next character is digit-like
            if feature_str.find_first_of(number_like, i) == i:
                str_begin = feature_str.c_str() + i
                parsed_val = strtod(str_begin, &str_end)
                if str_begin == str_end:
                    raise ValueError("Feature value is not a number: '{}'".format(feature_str.substr(i)))
                else:
                    partial.feat_val = parsed_val
                    inside_feature = 0
                    add_partial(e, partial, audit)
                    i += str_end - str_begin
            else:
                raise ValueError("Invalid feature declaration (after ':'). Rest of line: {}".format(feature_str.substr(i)))

        elif c == b'|':
            found = feature_str.find_first_of(space_or_colon, i)
            assert found != npos
            partial.ns_name = feature_str.substr(i + 1, found - i - 1)
            # print "found ns '{}'".format(partial.ns_name)
            i = found

            if feature_str[i] == b':':
                found = feature_str.find_first_of(space, i)
                assert found != npos
                parsed_val = strtod(feature_str.c_str() + i, NULL)
                if not parsed_val:
                    raise ValueError("Invalid namespace multiplier value")
                partial.ns_mult = parsed_val
                # print "ns multiplier", partial.ns_mult
                i = found
        elif c == b' ':
            # Skip space
            i += 1
        elif c == b'\n':
            i += 1
        else:
            # Feature
            found = feature_str.find_first_of(space_or_colon, i)
            assert found != npos
            # print "got feature", feature_str.substr(i, found - i)
            partial.feat_begin = i
            partial.feat_len = found - i
            partial.feat_val = 1.0
            if feature_str[found] == b':' and found < feature_str.size():
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
        string new_line = b'\n'

    label_map = {}
    if labels:
        for i, label_name in enumerate(labels):
            label_map[label_name] = i

    dataset = dataset_new(quadratic, ignore)
    seqs = []

    e = example_new(dataset)
    seq = Sequence()

    cfile = fopen(bytes(filename, encoding='utf-8'), b"rb")


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
        if line_str[line_str.size() - 1] != b'\n':
            line_str.push_back(b'\n')

        if line_str.size() > 1:
            e = example_new(dataset)

            bar_pos = line_str.find(b'|')

            if bar_pos == npos:
                raise ValueError("Missing | character in example")

            header = line_str.substr(0, bar_pos)
            parse_header(header, label_map, &e, audit)
            if require_labels and e.labels.size() == 0:
                raise ValueError("Missing label in example: {}".format(line))


            feature_section = line_str.substr(bar_pos)
            partial.feature_str = &feature_section
            parse_features2(&e, audit, &partial)

            # Add constant feature
            add_feature(&e, feat_map.feat_i(b"^Constant"), 1)

            seq.examples.push_back(e)

            if audit:
                print("")

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
