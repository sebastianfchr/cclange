import pandas as pd, numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold
from transformers import *
import tokenizers # transformers==3.0.2 with tf 2.7 apparently . Set rust path?
import models

print('TF version',tf.__version__)

# import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

MAX_LEN = 96

import utils

sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
    # pass vocab_path and merges_path

# make a tokenizer for encodings
tokenizer = tokenizers.ByteLevelBPETokenizer.from_file('config/vocab-roberta-base.json', 'config/merges-roberta-base.txt', lowercase=True, add_prefix_space=True) # !!!!!!! MATE! THIS SOLVES IT!

# get the token-sequences and the attention masks after tokenization
# the train-dataset additionally has "start_tokens, end_tokens", which are one-hot encoded positions of the range of tokens corresponding to a sentiment

test = pd.read_csv('data/test.csv').fillna('')
train = pd.read_csv('data/train.csv').fillna('')

input_ids, attention_mask, token_type_ids, start_tokens, end_tokens = utils.prepare_encode_train(train, MAX_LEN, tokenizer)
input_ids_t, attention_mask_t, token_type_ids_t = utils.prepare_encode_test(test, MAX_LEN, tokenizer)

# ================== Loss metric ==================
# will be used to determine overlap of tokenized sentence's token-subsequence from
# prediction (later rextracted per predicted one-hot indices) and desired subsequence
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    if (len(a)==0) & (len(b)==0): return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


# 1) functional API style
model = models.build_model(MAX_LEN) 
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# 2) alternatively, tf.keras.models.Model style, but has some disadvantages
# model = models.MyRobertaModel()
# optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# model.load_weights('weights_final.h5', by_name=True, skip_mismatch=False) # load HUK model
# print("apparently success")



# note that there's model.predict instead of the simple call to handle large inputs

# TODO: wrap around this... something that tokenizes a sentence

def predict_tokenized(input_ids, attention_mask, max_len_tokens : int):

    assert(input_ids.shape==attention_mask.shape 
           and len(input_ids.shape)==2 and input_ids.shape[1]==max_len_tokens) 
    
    # those aren't used by roberta, but required as input. Create here ...
    token_type_ids = tf.zeros(input_ids.shape, dtype=tf.int32)

    ls_one_hot, rs_one_hot = model(    
        (input_ids, attention_mask, token_type_ids)
    )

    # return ls_one_hot, rs_one_hot
    ls = np.argmax(ls_one_hot, -1)
    rs = np.argmax(rs_one_hot, -1)


    return ls, rs

# this is single. TODO: We can go further and chunk things inside this function
def predict_sentence(sentence: str, sentiment: str, tokenizer, max_len_tokens : int):

    assert(sentiment in utils.sentiment_id.keys())

    # this merges spaces, and adds a space at the front (if there's none already)
    text= " "+" ".join(sentence.split())
    enc = tokenizer.encode(text)        

    # TODO: currently 1. Change when chunking
    input_ids = np.ones((1,max_len_tokens),dtype='int32') # actually ones! 1:<pad> in vocab
    attention_mask = np.zeros((1, max_len_tokens), dtype='int32')
        
    # ///// s_tok = sentiment_id[test.loc[k,'sentiment']]

    # fill appropriately
    input_ids[0,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [utils.sentiment_id[sentiment]] + [2]
    attention_mask[0,:len(enc.ids)+5] = 1

    ls, rs = predict_tokenized(input_ids, attention_mask, max_len_tokens)

    l = ls[0]
    r = rs[0]
    print("l,r = ", l , r)

    st = tokenizer.decode(enc.ids[l-1:r])
 
    return st






# TODO: How exactly is the attention mask structured? 
# See Note: The attention-mask reveals exactly the parts of the sentence that's not void [0 cover only pads]


l, r = predict_tokenized(input_ids[0:3], attention_mask[0:3], MAX_LEN)

print(l)
print(r)

print(predict_sentence("water is wet", "positive", tokenizer, MAX_LEN))


exit(0)



# # ================== train ==================
jac = []; VER='v0'; DISPLAY=1 # USE display=1 FOR INTERACTIVE
oof_start = np.zeros((input_ids.shape[0],MAX_LEN))
oof_end = np.zeros((input_ids.shape[0],MAX_LEN))
preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))
preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))

skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=777)
for fold,(idxT,idxV) in enumerate(skf.split(input_ids, train.sentiment.values)):

    print('#'*25)
    print('### FOLD %i'%(fold+1))
    print('#'*25)
    
    # fresh model. I believe 
    K.clear_session()
    # model = build_model()
    # model.load_weights('weights_final.h5', by_name=True, skip_mismatch=False) # load HUK model

    # callback for saving
    sv = tf.keras.callbacks.ModelCheckpoint('%s-roberta-%i.h5'%(VER,fold), monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=True, mode='auto', save_freq='epoch')

    # model = tf.keras.models.load_model('weights_final.h5') # load HUK model
    # model.load_weights('weights_final.h5', by_name=True) # load HUK model
    model.fit([input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]], [start_tokens[idxT,], end_tokens[idxT,]], 
        epochs=3, batch_size=32  , verbose=DISPLAY, #callbacks=[sv],
        validation_data=([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]], 
        [start_tokens[idxV,], end_tokens[idxV,]]))
    
    print('Loading model...')
    model.load_weights('%s-roberta-%i.h5'%(VER,fold))
    
    print('Predicting OOF...')
    oof_start[idxV,],oof_end[idxV,] = model.predict([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]],verbose=DISPLAY)
    
    print('Predicting Test...')
    preds = model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)
    preds_start += preds[0]/skf.n_splits    # this looks like it should have a preds_start[idV, ]
    preds_end += preds[1]/skf.n_splits      # this looks like it should have a preds_end[idV, ]
    
    # DISPLAY FOLD JACCARD
    # this 
    all = []
    for k in idxV:
        a = np.argmax(oof_start[k,])
        b = np.argmax(oof_end[k,])
        if a>b: 
            st = train.loc[k,'text'] # IMPROVE CV/LB with better choice here
        else:
            text1 = " "+" ".join(train.loc[k,'text'].split())
            enc = tokenizer.encode(text1)
            st = tokenizer.decode(enc.ids[a-1:b])
        all.append(jaccard(st,train.loc[k,'selected_text']))
    jac.append(np.mean(all))
    print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all))
    print()