import json
import os
import subprocess
import pandas as pd

DEVNULL = open(os.devnull, 'wb')

test_sets= "gweb-answers-test gweb-emails-test gweb-newsgroups-test gweb-reviews-test gweb-weblogs-test ontonotes-wsj-test twittertest_gold_sd165".split(" ")

def labeler(options):
    cmd = "python rungsted/labeler.py {}".format(options)
    # print "Running: {}".format(cmd)
    subprocess.check_call(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL)


def print_results(results_file):
    objs = [json.loads(l) for l in open(results_file)]
    D = pd.DataFrame(objs)[['name', 'accuracy']]
    D = D.set_index('name')
    print D.to_string()
    print "Mean accuracy: {:.2%}".format(D.accuracy.mean())

def check_test_performance(model, tag_set):
    if os.path.exists("results.jsonl"):
        os.remove("results.jsonl")
    for test in test_sets:
        if test.startswith('twittertest_gold_sd165') and tag_set == 'coarse':
            # Not with coarse tags
            continue

        labeler("--initial-model {} --test data/{}.{}.vw --append-test results.jsonl --name {}".format(model, test, tag_set, test))
    print_results("results.jsonl")


# print "\nPlain (fine)"
# labeler("--train data/ontonotes-wsj-train.fine.vw --final-model models/ontonotes-wsj-train-plain-fine --labels data/labels.fine.txt --no-ada-grad --passes 10")
# check_test_performance("models/ontonotes-wsj-train-plain-fine", 'fine')
#
print "\nWith drop-out (fine)"
labeler("--train data/ontonotes-wsj-train.fine.vw --final-model models/ontonotes-wsj-train-drop_out-fine --labels data/labels.fine.txt --no-ada-grad --passes 10 --drop-out")
check_test_performance("models/ontonotes-wsj-train-drop_out-fine", 'fine')

# print "\nPlain (coarse)"
# labeler("--train data/ontonotes-wsj-train.coarse.vw --final-model models/ontonotes-wsj-train-plain-coarse --passes 10 --no-ada-grad")
# check_test_performance("models/ontonotes-wsj-train-plain-coarse", 'coarse')

print "\nWith drop-out (coarse)"
labeler("--train data/ontonotes-wsj-train.coarse.vw --final-model models/ontonotes-wsj-train-drop_out-coarse --no-ada-grad --passes 10 --drop-out")
check_test_performance("models/ontonotes-wsj-train-drop_out-coarse", 'coarse')



# print "\nWithout ada-grad"
# labeler("--train data/ontonotes-wsj-train.vw --final-model models/ontonotes-wsj-train --labels data/labels.fine.txt --no-ada-grad")
# check_test_performance("models/ontonotes-wsj-train")
#
# print "\nWithout average"
# labeler("--train data/ontonotes-wsj-train.vw --final-model models/ontonotes-wsj-train --labels data/labels.fine.txt --no-average")
# check_test_performance("models/ontonotes-wsj-train")
#
# print "\nWith 19 bit hashing"
# labeler("--train data/ontonotes-wsj-train.vw --final-model models/ontonotes-wsj-train --labels data/labels.fine.txt -b 19")
# check_test_performance("models/ontonotes-wsj-train")
