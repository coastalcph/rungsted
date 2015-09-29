[![Build Status](https://travis-ci.org/coastalcph/rungsted.svg?branch=master)](https://travis-ci.org/coastalcph/rungsted)


## Rungsted structured perceptron sequential tagger

### Install

The software is installable via PyPI, e.g. do 

```
pip install rungsted
```



### Demo

The repository contains a subset of the part-of-speech tagged Brown corpus. To run the structured perceptron labeler on this dataset, execute:

``python src/labeler.py --train data/brown.train --test data/brown.test.vw``

Rungsted's input format is closely modeled on the powerful and flexible format of [Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit/wiki/Input-format),
with the exception that Rungsted is perfectly fine with labels that are not integers.

### Datasets

Provided you have a working installation of NLTK, you can recreate the Brown dataset with this command. 

``python rungsted/datasets/cr_brown_pos_data.py data/brown.train.vw data/brown.test.vw``

There is also a script `rungsted/datasets/conll_to_vw.py` to convert from CONLL-formatted input to Rungsted 


### Building and uploading to PyPI

First, run `python setup.py sdist` to generate a source distribution. 
Then upload the distribution files to PyPI with twine: `twine upload dist/*`.

To develop locally, use `python setup.py develop`. 
