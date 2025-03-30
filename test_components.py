
from roberta import RobertaPredictor, TokenEncoder
import tokenizers
import pandas as pd
import pytest 

MAX_LEN = 96
        
class TestClassPredictor: 

    def test_batched_sentence_extraction_vs_manual(self):
        """ Sanity check for RobertaPredictor.predict_sentence_batch(). Tests whether it produces 
        the same sentence-fragment as manual tokenization, prediction, and decoding """

        num_elements_tested = 10

        tokenizer = tokenizers.ByteLevelBPETokenizer.from_file('./roberta/config/vocab-roberta-base.json', './roberta/config/merges-roberta-base.txt', lowercase=True, add_prefix_space=True) 
        rp = RobertaPredictor(MAX_LEN, 'v0-roberta-0.h5', tokenizer)

        # Version 1) 
        # Use inputs from tokenization, manual tokenized prediction, and decoding of prediction to sentence-fragments

        test = pd.read_csv('./data/test.csv').fillna('')
        e = TokenEncoder(MAX_LEN, tokenizer)
        input_ids_t, attention_mask_t, _ = e.prepare_encode_test(test)
        # ids and attention_masks. Predict left end right end of subsequence-range within ids 
        ids, ams = input_ids_t[0:num_elements_tested], attention_mask_t[0:num_elements_tested]
        ls, rs= rp.predict_tokenized(ids, ams)
        # Based on (l, r), obtain the extracted ranges        
        predicted_subsequences = [ids[i, l:r+1] for i, (l,r) in enumerate(zip(ls,rs))]
        # decode them into sentence sub-fragments via tokenizer
        sentences_manual = list(map(tokenizer.decode, predicted_subsequences))


        # Version 2) 
        # Encapsulated tokenization, prediction, decoding of plain-text inside function

        # now note that "ids" were already (<s> {sentence_tokenized} </s></s> sentiment_token </s>)
        # if we feed in (sentence, sentiment) as strings, our RobertaPredictor.predict_sentence_batch 
        sentences_automatic = rp.predict_sentence_batch(test['text'][0:num_elements_tested], test['sentiment'][0:num_elements_tested])


        # 1) and 2) should have the same result        
        assert(sentences_manual == sentences_automatic)

class TestAPI:
    """ These tests make sure that the API behaves properly """

    def test_single_request(self):
        pass

    def test_batch_request(self):
        pass 