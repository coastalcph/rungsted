#/bin/bash
for file in struct_perceptron input feat_map decoding decoding_pd weights; do
    rm -f rungsted/$file.cpp
done;