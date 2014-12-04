#!/bin/sh
IN_DIR=~/data/treebanks/english

for file in $IN_DIR/*.conll; do
    name=$(basename $file)
    name=${name%.conll}
    echo "Processing $name"
    python rungsted/datasets/conll_to_vw.py $file data/$name.fine.vw --name $name --feature-set honnibal13-groups
    python rungsted/datasets/conll_to_vw.py $file data/$name.coarse.vw --name $name --feature-set honnibal13-groups --coarse

done;
