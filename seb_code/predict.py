import tensorflow as tf
import models
import numpy as np
import utils

class RobertaPredictor:
    
    def __init__(self, max_len_tokens: int, weights_path: str, tokenizer):
        # Load structure, then weights. Weights must be it's structurally correct
        self.max_len_tokens = max_len_tokens
        self.model = models.build_model(self.max_len_tokens)
        self.model.load_weights(weights_path, by_name=True, skip_mismatch=False) 
        self.tokenizer = tokenizer


    # TODO: @tf.graph
    def predict_tokenized(self, input_ids, attention_mask):
        # Batch-wise prediction of token-substrings encoded in input_ids and attention_masks,
        # whereas predicted substrings conform to the sentiment. Note that the sentiment is 
        # already part of the input_ids (see also predict_sentence, or original notebook)

        assert(input_ids.shape==attention_mask.shape 
            and len(input_ids.shape)==2 and input_ids.shape[1]==self.max_len_tokens)
        
        # those aren't used by roberta, but required as input. Create here ...
        token_type_ids = tf.zeros(input_ids.shape, dtype=tf.int32)

        ls_one_hot, rs_one_hot = self.model(
            (input_ids, attention_mask, token_type_ids)
        ) # both (?, max_len_tokens)

        # return maximum index by using argmax on last dim 
        ls = np.argmax(ls_one_hot, -1)
        rs = np.argmax(rs_one_hot, -1)

        return ls, rs


    def predict_sentence(self, sentence: str, sentiment: str):

        assert(sentiment in utils.sentiment_id.keys())

        # this merges spaces, and adds a space at the front (if there's none already)
        sentence_prepared= " "+" ".join(sentence.split())
        enc = self.tokenizer.encode(sentence_prepared)        

        # TODO: currently 1. Change when chunking
        input_ids = np.ones((1,self.max_len_tokens),dtype='int32') # actually ones! 1:<pad> in vocab
        attention_mask = np.zeros((1, self.max_len_tokens), dtype='int32')

        print(utils.sentiment_id[sentiment])

        # fill appropriately
        input_ids[0,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [utils.sentiment_id[sentiment]] + [2]
        attention_mask[0,:len(enc.ids)+5] = 1

        ls, rs = self.predict_tokenized(input_ids, attention_mask)

        l = ls[0]
        r = rs[0]
        print("l,r = ", l , r)

        print(enc.ids)

        st = self.tokenizer.decode(input_ids[0, l:r+1])

        return st
    
    
    def predict_sentence_batch(self, sentences: list[str], sentiments: list[str]):
        # writing is cumbersome, but optimized: allow batchwise prediction inside model!

        assert(all([s in utils.sentiment_id.keys() for s in sentiments]))
        assert(len(sentences) == len(sentiments))
        n = len(sentences)

        # Per sentence: merges spaces, and adds a space at the front, then encode
        sentences_prepared = list(map(lambda stc:  " "+" ".join(stc.split()), sentences ))
        encs = list(map(self.tokenizer.encode, sentences_prepared))

        input_idss = np.ones((n, self.max_len_tokens),dtype='int32') # actually ones! 1:<pad> in vocab
        attention_masks = np.zeros((n, self.max_len_tokens), dtype='int32')

        for i in range(n):
            input_idss[i, 0:len(encs[i].ids)+5] = [0] + encs[i].ids + [2,2] + [utils.sentiment_id[sentiments[i]]] + [2]
            attention_masks[i, 0:len(encs[i].ids)+5] = 1

        # here's the reason we did all this: efficient call of model!
        ls, rs = self.predict_tokenized(input_idss, attention_masks)
        predicted_subsequences = [input_idss[i, l:r+1] for i, (l,r) in enumerate(zip(ls,rs))]

        sts = list(map(self.tokenizer.decode, predicted_subsequences))

        return sts
    
    


# TODO: move somewhere else
import tokenizers
import utils
import pandas as pd

MAX_LEN = 96

sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}

# make a tokenizer for encodings
tokenizer = tokenizers.ByteLevelBPETokenizer.from_file('config/vocab-roberta-base.json', 'config/merges-roberta-base.txt', lowercase=True, add_prefix_space=True) # !!!!!!! MATE! THIS SOLVES IT!

test = pd.read_csv('data/test.csv').fillna('')
train = pd.read_csv('data/train.csv').fillna('')
input_ids, attention_mask, token_type_ids, start_tokens, end_tokens = utils.prepare_encode_train(train, MAX_LEN, tokenizer)
input_ids_t, attention_mask_t, token_type_ids_t = utils.prepare_encode_test(test, MAX_LEN, tokenizer)


rp = RobertaPredictor(MAX_LEN, '/home/seb/Desktop/CodingChallenge_MLE/v0-roberta-0.h5', tokenizer)
# rp = RobertaPredictor(MAX_LEN, '/home/seb/Desktop/CodingChallenge_MLE/seb_code/weights_final.h5')
# TODO: This should be the test. Tokenized predicts same 

ids, ams = input_ids[0:10], attention_mask[0:10]
ls, rs= rp.predict_tokenized(ids, ams)

# print(ids)
print(ls, rs)
predicted_subsequences = [ids[i, l:r+1] for i, (l,r) in enumerate(zip(ls,rs))]
print(predicted_subsequences)
sentences = list(map(tokenizer.decode, predicted_subsequences))
print("\n".join(sentences))
# tokeni 
# print(predicted_subsequences)

print("==============================")
print("==============================")

# print(rp.predict_sentence("my boss is bullying me...","negative"))
print(rp.predict_sentence_batch(["my boss is bullying me..."],["negative"]))

# print(tokenizer.encode(" the big bad wolf").ids)
exit(0)


