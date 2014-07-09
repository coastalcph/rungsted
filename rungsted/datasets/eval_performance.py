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

def check_test_performance(model):
    if os.path.exists("results.jsonl"):
        os.remove("results.jsonl")
    for test in test_sets:
        labeler("--initial-model {} --test data/{}.vw --append-test results.jsonl --name {}".format(model, test, test))
    print_results("results.jsonl")


print "\nWith ada-grad"
labeler("--train data/ontonotes-wsj-train.vw --final-model models/ontonotes-wsj-train --labels data/labels.fine.txt")
check_test_performance("models/ontonotes-wsj-train")

print "\nWithout ada-grad"
labeler("--train data/ontonotes-wsj-train.vw --final-model models/ontonotes-wsj-train --labels data/labels.fine.txt --no-ada-grad")
check_test_performance("models/ontonotes-wsj-train")

print "\nWithout average"
labeler("--train data/ontonotes-wsj-train.vw --final-model models/ontonotes-wsj-train --labels data/labels.fine.txt --no-average")
check_test_performance("models/ontonotes-wsj-train")

print "\nWith 19 bit hashing"
labeler("--train data/ontonotes-wsj-train.vw --final-model models/ontonotes-wsj-train --labels data/labels.fine.txt -b 19")
check_test_performance("models/ontonotes-wsj-train")
