import tensorflow as tf
import numpy as np
from . import models
from . import utils
import tokenizers

# TODO: Naming. Technically, not sentences.


class RobertaPredictor:


    def __init__(self, 
                 max_len_tokens : int = 96, 
                 weights_path : str ='v0-roberta-0.h5', 
                 tokenizer = tokenizers.ByteLevelBPETokenizer.from_file('config/vocab-roberta-base.json', 'config/merges-roberta-base.txt', lowercase=True, add_prefix_space=True)):
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

    
    def predict_sentence_batch(self, sentences: list[str], sentiments: list[str]):
        # writing is cumbersome, but optimized: allow batchwise prediction inside model!

        assert(all([s in utils.sentiment_id.keys() for s in sentiments]))
        assert(len(sentences) == len(sentiments))
        n = len(sentences)

        # Per sentence: merges spaces, and adds a space at the front, then encode
        sentences_prepared = map(lambda stc:  " "+" ".join(stc.split()), sentences)
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
    

    def predict_sentence(self, sentence: str, sentiment: str):
        # single prediction is just a sentence-batch of size 1
        return self.predict_sentence_batch([sentence], [sentiment])



    
    