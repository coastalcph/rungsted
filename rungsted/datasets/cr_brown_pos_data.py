import argparse
import codecs
from collections import Counter, defaultdict
from itertools import islice, count
import logging
from nltk.corpus import brown

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(description="Create a dataset for part-of-speech tagging on the brown corpus, "
                                             "which can be feed to Vowpal Wabbit compatible software")
parser.add_argument('outfile', help="Place generated dataset here")
parser.add_argument('class_map', help="Write a mapping from string-valued labels to integer labels")

args = parser.parse_args()

sents = list(islice(brown.tagged_sents(simplify_tags=True), None))
# train = sents[0:800]
# test = sents[800:1000]
label_counter = count(1)
label2id = defaultdict(label_counter.next)

data_out = codecs.open(args.outfile, 'w', encoding='utf-8')

def normalize_word(word):
    return word.replace(":", "COL")

# Features from http://www.seas.upenn.edu/~taskar/pubs/wikipos_emnlp12.pdf
#  Word identity - lowercased word form if the word appears more than 10 times in the corpus.
#  Hyphen - word contains a hyphen
#  Capital - word is uppercased
#  Suffix - last 2 and 3 letters of a word if they appear in more than 20 different word types.
#  Number - word contains a digit

all_words = [normalize_word(word.lower()) for sent in sents for word, label in sent]
word_counts = Counter(all_words)
word_whitelist = set(word for word, count in word_counts.most_common() if count > 10)

words_by_suffix = defaultdict(set)
for word in all_words:
    words_by_suffix[word[-2:]].add(word)
    words_by_suffix[word[-3:]].add(word)
suffix_whitelist = set(suffix for suffix, word_set in words_by_suffix.items() if len(word_set) > 20)

for sent_i, sent in enumerate(sents):
    for word_i, (word, label) in enumerate(sent):
        if label == '':
            continue
        # assert tag != '', "Tag for {} is empty".format(word)
        feats = []
        norm = normalize_word(word.lower())
        if norm in word_whitelist:
            feats.append("w={}".format(norm))
        if norm[-2:] in suffix_whitelist:
            feats.append("suf2={}".format(norm[-2:]))
        if norm[-3:] in suffix_whitelist:
            feats.append("suf3={}".format(norm[-3:]))
        if word[0].isupper():
            feats.append("uppercased")
        if any(map(str.isdigit, norm)):
            feats.append("hasdigit")
        print >>data_out, "{label} '{id}| {feats}".format(label=label2id[label],
                                              id="{}-{}".format(sent_i, word_i),
                                              feats=" ".join(feats))
    print >>data_out, ""

data_out.close()

with codecs.open(args.class_map, 'w', encoding='utf-8') as out:
    for key in sorted(label2id.keys()):
        print >>out, "{}\t{}".format(label2id[key], key)
