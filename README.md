## Rungsted structed perceptron sequential tagger

### Building 

Use

``python setup.py build_ext --inplace``

Building with the above command happens *in place*, leaving the generated C and C++ files in the source directory for inspection. Changes in dependent modules are unfortunately not picked up by the build system. Whenever you need to start from a clean slate, use the supplied `clean.sh` script to get rid of the generated files. 

### Demo

The repository contains a subset of the part-of-speech tagged Brown corpus. To run the structured perceptron labeler on this dataset, use:

``python src/runner.py --train data/brown.train --test data/brown.test.vw -k 39``

Labels must be integers in the range 1..k. The *k* parameter is thus the number of distinct labels in the input data.

## Usage


```
usage: runner.py [-h] [--train TRAIN] [--test TEST] [--hash-bits HASH_BITS]
                 --n-labels N_LABELS [--passes PASSES]
                 [--predictions PREDICTIONS] [--ignore [IGNORE [IGNORE ...]]]
                 [--decay-exp DECAY_EXP] [--decay-delay DECAY_DELAY]
                 [--shuffle] [--average]

Structured perceptron tagger

optional arguments:
  -h, --help            show this help message and exit
  --train TRAIN         Training data (vw format)
  --test TEST           Test data (vw format)
  --hash-bits HASH_BITS, -b HASH_BITS
                        Size of feature vector in bits (2**b)
  --n-labels N_LABELS, -k N_LABELS
                        Number of different labels
  --passes PASSES       Number of passes over the training set
  --predictions PREDICTIONS, -p PREDICTIONS
                        File for outputting predictions
  --ignore [IGNORE [IGNORE ...]]
                        One-character prefix of namespaces to ignore
  --decay-exp DECAY_EXP
                        Learning rate decay exponent. Learning rate is
                        (iteration no)^decay_exponent
  --decay-delay DECAY_DELAY
                        Delay decaying the learning rate for this many
                        iterations
  --shuffle             Shuffle examples after each iteration
  --average             Average over all updates

```
