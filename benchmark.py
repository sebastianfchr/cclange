import argparse
import requests
import pandas as pd
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from roberta import RobertaPredictor
import tokenizers
from typing import List

# parser = argparse.ArgumentParser("benchmarkparser")
# parser.add_argument("--reqserver", help="A server whose <<server>>/predict_sentence/ and <<server>>/predict_sentence_batch/ will receive POST Requests for benchmarking", type=str)
# args = parser.parse_args()
# print(args.reqserver)
# TODO: Parse!

url = "http://0.0.0.0:8123/predict_sentence_batch/"

# let's first take some data from the test-set
test = pd.read_csv('./data/test.csv').fillna('')


def single_server_requests(sentences: List[str], sentiments: List[str]):
    """ send server requests in the list one-by-one """
    start = time.time()
    for sentence, sentiment in zip(sentences, sentiments):
        data = {"sentences": [sentence], "sentiments": [sentiment]}
        response = requests.post(url, json=data, headers={ 'Content-Type': 'application/json' })
    return time.time()-start 

def chunked_server_requests(sentences: List[str], sentiments: List[str]):
    """ send server requests in the list one-by-one """
    start = time.time()
    data = {"sentences": [s for s in sentences], "sentiments": [s for s in sentiments]}
    response = requests.post(url, json=data, headers={ 'Content-Type': 'application/json' })
    # print(response.text)
    return time.time()-start


tokenizer = tokenizers.ByteLevelBPETokenizer.from_file('./roberta/config/vocab-roberta-base.json', './roberta/config/merges-roberta-base.txt', lowercase=True, add_prefix_space=True) 
rp = RobertaPredictor(96, 'v0-roberta-0.h5', tokenizer)

def single_bare_requests(sentences: List[str], sentiments: List[str]):
    """ direkt tf-model requests in the list one-by-one """
    start = time.time()
    for sentence, sentiment in zip(sentences, sentiments):
        rp.predict_sentence_batch([sentence], [sentiment])
    return time.time()-start 

def chunked_bare_requests(sentences: List[str], sentiments: List[str]):
    """ direkt tf-model requests in the list one-by-one """
    start = time.time()
    rp.predict_sentence_batch(sentences, sentiments)
    return time.time()-start 


xs = np.arange(1, 31, dtype=int)

# make one dummy request to warm up tensorflow cuda-kernel launch (I'm not joking!)
single_bare_requests(test['text'][:1], test['sentiment'][:1])
                     
ys_server_single = [single_server_requests(test['text'][:i], test['sentiment'][:i]) for i in xs]
ys_server_chunked = [chunked_server_requests(test['text'][:i], test['sentiment'][:i]) for i in xs]
ys_bare_single = [single_bare_requests(test['text'][:i], test['sentiment'][:i]) for i in xs]
ys_bare_chunked = [chunked_bare_requests(test['text'][:i], test['sentiment'][:i]) for i in xs]

# formatting and styling
fig, (p1, p2) = plt.subplots(1,2, figsize=(15,8), sharey=True)
p1.plot(xs, ys_server_single, '-*', label='single')
p1.plot(xs, ys_server_chunked, '-*', label='chunked')
p1.set_title("Direct TF calls", fontsize=18)
p2.plot(xs, ys_bare_single, '-*', label='single')
p2.plot(xs, ys_bare_chunked, '-*', label='chunked')
p2.set_title("API requests", fontsize=18)
p2.set_ylim(0, 3)

for p in [p1, p2]:
    p.set_xticks([1,5,10,15,20,25,30])
    p.set_xlabel('num predictions', fontsize=16)
p1.set_ylabel('overall time', fontsize=16)

plt.subplots_adjust(wspace=0.05)

p1.legend(fontsize=15)
p2.legend(fontsize=15)
fig.savefig('benchmark_tf_and_api_calls.png')

