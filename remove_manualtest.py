
from roberta import RobertaPredictor, TokenEncoder

# TODO: move somewhere else
import tokenizers
import pandas as pd

MAX_LEN = 96

# sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}

# make a tokenizer for encodings
# TODO: Tokenizer!
tokenizer = tokenizers.ByteLevelBPETokenizer.from_file('roberta/config/vocab-roberta-base.json', 'roberta/config/merges-roberta-base.txt', lowercase=True, add_prefix_space=True) # !!!!!!! MATE! THIS SOLVES IT!

test = pd.read_csv('./data/test.csv').fillna('')
train = pd.read_csv('./data/train.csv').fillna('')

e = TokenEncoder(MAX_LEN, tokenizer)
input_ids, attention_mask, token_type_ids, start_tokens, end_tokens = e.prepare_encode_train(train)
input_ids_t, attention_mask_t, token_type_ids_t = e.prepare_encode_test(test)

rp = RobertaPredictor(MAX_LEN, 'v0-roberta-0.h5', tokenizer)


# This should show us if the predict_sentence_batch works properly

# VERSION 1: tokenize everything manually, predict_tokenized, and then decode
ids, ams = input_ids[0:10], attention_mask[0:10]
ls, rs= rp.predict_tokenized(ids, ams)
print(ls, rs)
predicted_subsequences = [ids[i, l:r+1] for i, (l,r) in enumerate(zip(ls,rs))]
print(predicted_subsequences)
sentences = list(map(tokenizer.decode, predicted_subsequences))
print("\n".join(sentences))



print("==============================")
print("NOTE: Those two versions must be the same")
print("==============================")


# print("\n".join(rp.predict_sentence_batch(train['text'][0:10],train['sentiment'][0:10])))
print("============")
# print("\n".join(rp.predict_sentence_batch(test['text'][0:100],test['sentiment'][0:100])))
print("\n".join(rp.predict_sentence_batch(test['text'][0:20],test['sentiment'][0:20])))
