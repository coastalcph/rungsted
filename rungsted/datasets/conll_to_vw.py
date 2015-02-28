# coding: utf-8
"""Create Rungsted compatible feature files from treebanks

Usage:
  conll_to_vw.py <input> <output> [--feature-set NAME] [--name NAME] [--coarse]
  conll_to_vw.py (-h | --help)

Options:
  -h --help                 Show this screen.
  --feature-set NAME        Which feature to use [default: honnibal13].
  --name NAME               Name of dataset. Output as part of the id for each token [default: d].
  --coarse                  Use coarse-grained tags.
"""
import codecs
from collections import defaultdict
from docopt import docopt
from pos_features import taskar12, honnibal13, honnibal13_groups, normalize_label, normalize_word

if __name__ == '__main__':
    args = docopt(__doc__)

    data_out = codecs.open(args['<output>'], 'w', encoding='utf-8')
    data_in = codecs.open(args['<input>'], encoding='utf-8')

    features_for_token = {
        'taskar12': taskar12,
        'honnibal13': honnibal13,
        'honnibal13-groups': honnibal13_groups
    }[args['--feature-set']]

    def output_sentence(sent):
        label_key = 'cpos' if args['--coarse'] else 'pos'
        for i in range(len(sent['word'])):
            print >>data_out, "{label} '{name}-{sent_i}-{token_i}|".format(
                label=normalize_label(sent[label_key][i]),
                name=args['--name'],
                sent_i=sent_i,
                token_i=i+1),
            print >>data_out, u" ".join(features_for_token(sent['word'], sent[label_key], i))

    # Process one sentence at a time
    sent = defaultdict(list)
    sent_i = 1
    for line in data_in:
        parts = line.strip().split()
        if len(parts) == 10:
            word = parts[1]
            if word.isdigit():
                number = int(word)
                if 1800 <= number <= 2100:
                    word = "!YEAR"
                else:
                    word = "!DIGITS"

            sent['word'].append(word)
            sent['cpos'].append(parts[3])
            sent['pos'].append(parts[4])
        elif len(parts) == 0:
            if sent_i > 1:
                print >>data_out, ""
            output_sentence(sent)
            sent_i += 1
            sent = defaultdict(list)
        else:
            raise "Invalid input format"

    if len(sent['word']):
        output_sentence(sent)