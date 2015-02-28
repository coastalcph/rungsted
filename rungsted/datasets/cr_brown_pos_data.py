import argparse
import codecs
import logging
import random

from nltk.corpus import brown

from pos_features import honnibal13, normalize_label


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(description="Extract the Brown corpus from NLTK, "
                                             "writing it into Rungsted compatible feature files.")
parser.add_argument('train_file', help="Place generated dataset here")
parser.add_argument('test_file', help="Place generated dataset here")

args = parser.parse_args()


def write_sentences_to_file(sents, data_out):
    for sent_i, sent in enumerate(sents):
        tokens = [token for token, pos in sent]
        tags = [pos for token, pos in sent]

        if sent_i > 0:
            print >> data_out, ""

        for token_i in range(len(sent)):
            print >> data_out, "{label} '{name}-{sent_i}-{token_i}|".format(
                label=normalize_label(tags[token_i]),
                name='brown',
                sent_i=sent_i,
                token_i=token_i),

            print >> data_out, u" ".join(honnibal13(tokens, tags, token_i))


sents = list(brown.tagged_sents(simplify_tags=True))

# Create a random training and test set
random.seed(42)
random.shuffle(sents)
train = sents[1000:5000]
test = sents[:500]

train_out = codecs.open(args.train_file, 'w', encoding='utf-8')
test_out = codecs.open(args.test_file, 'w', encoding='utf-8')

write_sentences_to_file(train, train_out)
write_sentences_to_file(test, test_out)