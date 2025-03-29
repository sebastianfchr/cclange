import numpy as np
from transformers import *

# this is the assumption
# TODO: Maybe make Encoder!
sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}

def prepare_encode_train(train_pd, max_len_tokens, tokenizer):

    # ================================== train ======================================== 
    num_train_entries = train_pd.shape[0]
    input_ids = np.ones((num_train_entries,max_len_tokens),dtype='int32')
    attention_mask = np.zeros((num_train_entries,max_len_tokens),dtype='int32')
    token_type_ids = np.zeros((num_train_entries,max_len_tokens),dtype='int32')
    start_tokens = np.zeros((num_train_entries,max_len_tokens),dtype='int32')
    end_tokens = np.zeros((num_train_entries,max_len_tokens),dtype='int32')


    for k in range(train_pd.shape[0]):
        
        # FIND OVERLAP
        text1 = " "+" ".join(train_pd.loc[k,'text'].split())
        text2 = " ".join(train_pd.loc[k,'selected_text'].split())
        idx = text1.find(text2)
        chars = np.zeros((len(text1)))
        chars[idx:idx+len(text2)]=1                             # character-position mask = 1 for text2 inside text1 
        if text1[idx-1]==' ': chars[idx-1] = 1                  # space before text2 is also included in mask
        enc = tokenizer.encode(text1) 
            
        # ID_OFFSETS
        offsets = []; idx=0
        # 'offsets' are the string's subranges that correspond to the encoded tokens enc.id
        # in congruent sequence. (Ranges (left, right) not including 'right')
        # this maps enc.ids back onto their respective string-ranges in text1
        for t in enc.ids:
            w = tokenizer.decode([t])
            offsets.append((idx,idx+len(w)))
            idx += len(w)
        
        # print("offsets", offsets)
        # START END TOKENS
        toks = []
        # remember that 'enc.ids' is a tokenization of text1, and that text2 ⊆ text1
        # 'toks' is an index-array on enc.ids, which indicates tokens text2 within text1 
        for i,(a,b) in enumerate(offsets): 
            sm = np.sum(chars[a:b])
            if sm>0: toks.append(i) 

        # print(toks)

            
        s_tok = sentiment_id[train_pd.loc[k,'sentiment']]
        # this one does 0{text..}22{sentiment}2
        
        # input_ids encodes the sequence of tokens corresponding to text1, plus the sentiment 
        # of the to-be cropped subsequence. Those are encoded as token sequences with start and end token,
        # i.e. <s> {...tokens} </s></s> belonging_sentiment_id </s> ["</s></s>" seems counter-intuitive, but original online version has it too]
        input_ids[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]
        # Note: The attention-mask reveals exactly the parts of the sentence that's not void [0 cover only pads]
        attention_mask[k,:len(enc.ids)+5] = 1

        # remember that 'toks' encode the indices of text2-tokens within of the tokenized text1-string
        # here, we encode the first and last position of 'toks' as one-hot [in start_tokens and end_tokens] 
        # as ground-truth for the beginning and end indices of post-tokenized text2 within text1 (i.e. toks[0], toks[-1] to one-hot vectors)
        # => this will be predicted from (tokenized_text1, tokenized_sentiment) 
        if len(toks)>0:
            # print("aaa")
            start_tokens[k,toks[0]+1] = 1
            end_tokens[k,toks[-1]+1] = 1


    return input_ids, attention_mask, token_type_ids, start_tokens, end_tokens



def prepare_encode_test(test_pd, max_len_tokens, tokenizer):
    # ================== Test data preparation ==================
    # Similar for test, as we did for train. Here, we only have token-sequences and sentiments
    # (that means, since nothing is extracted, we'll just check whether the sentiment is correct?)
    # Q: why exactly is the test-data only [(extracted/original ?)sequence, sentiment] enough for testing? 
    #    i.e. Why is extraction not tested ?

    num_test_entries = test_pd.shape[0]
    input_ids_t = np.ones((num_test_entries, max_len_tokens),dtype='int32')
    attention_mask_t = np.zeros((num_test_entries, max_len_tokens),dtype='int32')
    token_type_ids_t = np.zeros((num_test_entries, max_len_tokens),dtype='int32') # NOTE: NOT used in RoBERTa, and probably don't help us either


    for k in range(test_pd.shape[0]):
            
        # INPUT_IDS
        text1 = " "+" ".join(test_pd.loc[k,'text'].split())
        enc = tokenizer.encode(text1)
                    
        s_tok = sentiment_id[test_pd.loc[k,'sentiment']]
        input_ids_t[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]
        attention_mask_t[k,:len(enc.ids)+5] = 1

    return input_ids_t, attention_mask_t, token_type_ids_t



# ================== Loss metric ==================
# will be used to determine overlap of tokenized sentence's token-subsequence from
# prediction (later rextracted per predicted one-hot indices) and desired subsequence
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    if (len(a)==0) & (len(b)==0): return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

