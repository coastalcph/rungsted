## Rungsted structed perceptron sequential tagger

### Building 

Use

``python setup.py build_ext --inplace``

Building with the above command happens *in place*, leaving the generated C and C++ files in the source directory for inspection. Changes in dependent modules are unfortunately not picked up by the build system. Whenever you need to start from a clean slate, use the supplied `clean.sh` script to get rid of the generated files. 

### Demo

The repository contains a subset of the part-of-speech tagged Brown corpus. To run the structured perceptron labeler on this dataset, use:

``python src/labeler.py --train data/brown.train --test data/brown.test.vw``

## Usage


```
usage: labeler.py [-h] [--train TRAIN] [--test TEST] [--hash-bits HASH_BITS]
                  [--passes PASSES] [--predictions PREDICTIONS]
                  [--ignore [IGNORE [IGNORE ...]]]
                  [--quadratic [QUADRATIC [QUADRATIC ...]]] [--shuffle]
                  [--no-average] [--initial-model INITIAL_MODEL]
                  [--final-model FINAL_MODEL] [--cost-sensitive] [--audit]

Structured perceptron tagger

optional arguments:
  -h, --help            show this help message and exit
  --train TRAIN         Training data (vw format)
  --test TEST           Test data (vw format)
  --hash-bits HASH_BITS, -b HASH_BITS
                        Size of feature vector in bits (2**b)
  --passes PASSES       Number of passes over the training set
  --predictions PREDICTIONS, -p PREDICTIONS
                        File for outputting predictions
  --ignore [IGNORE [IGNORE ...]]
                        One-character prefix of namespaces to ignore
  --quadratic [QUADRATIC [QUADRATIC ...]], -q [QUADRATIC [QUADRATIC ...]]
                        Combine features in these two namespace, identified by
                        a one-character prefix of their name':' is a short-
                        hand for all namespaces
  --shuffle             Shuffle examples after each iteration
  --no-average          Do not average over all updates
  --initial-model INITIAL_MODEL, -i INITIAL_MODEL
                        Initial model from this file
  --final-model FINAL_MODEL, -f FINAL_MODEL
                        Save model here after training
  --cost-sensitive, --cs
                        Cost-sensitive weight updates
  --audit               Print the interpretation of the input files to
                        standard out. Useful for debugging
```
