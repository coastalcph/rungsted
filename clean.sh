#/bin/bash
for file in corruption struct_perceptron input feat_map decoding decoding_pd weights; do
    rm -f rungsted/$file.cpp
done;