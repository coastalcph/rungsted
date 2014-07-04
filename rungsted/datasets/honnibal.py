# Feature model from http://honnibal.wordpress.com/2013/09/11/a-good-part-of-speechpos-tagger-in-about-200-lines-of-python/

def _get_features(self, i, word, context, prev, prev2):
    '''Map tokens-in-contexts into a feature representation, implemented as a
    set. If the features change, a new model must be trained.'''
    feats = []

# - All words are lower cased;
# - Digits in the range 1800-2100 are represented as !YEAR;
# - Other digit strings are represented as !DIGITS


    def add(name, *args):
        features.add('+'.join((name,) + tuple(args)))

    # features = set()
    # add('bias') # This acts sort of like a prior
    # add('i suffix', word[-3:])
    # add('i pref1', word[0])
    # add('i-1 tag', prev)
    # add('i-2 tag', prev2)
    # add('i tag+i-2 tag', prev, prev2)
    # add('i word', context[i])
    # add('i-1 tag+i word', prev, context[i])
    # add('i-1 word', context[i-1])
    # add('i-1 suffix', context[i-1][-3:])
    # add('i-2 word', context[i-2])
    # add('i+1 word', context[i+1])
    # add('i+1 suffix', context[i+1][-3:])
    # add('i+2 word', context[i+2])
    # return features