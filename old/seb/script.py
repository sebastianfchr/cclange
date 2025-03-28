import pandas as pd, numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold
from transformers import *
import tokenizers # transformers==3.0.2 with tf 2.7 apparently . Set rust path?
print('TF version',tf.__version__)

# physical_devices = tf.config.list_physical_devices('GPU')
# print(physical_devices)
# try:
#   tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#   print("cant")
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

MAX_LEN = 96
PATH = './config/'

vocap_pth = PATH+'vocab-roberta-base.json'
merges_pth = PATH+'merges-roberta-base.txt'

# # older versions do it this way:    
# tokenizer = tokenizers.ByteLevelBPETokenizer(
#     vocab_file=vocap_pth,
#     merges_file=merges_pth,
#     lowercase=True,
#     add_prefix_space=True
# )
# tokenizer = tokenizers.ByteLevelBPETokenizer(
#     lowercase=True,
#     add_prefix_space=True
# )
# this makes it equal!
tokenizer = tokenizers.ByteLevelBPETokenizer.from_file(vocap_pth, merges_pth, lowercase=True, add_prefix_space=True) # !!!!!!! MATE! THIS SOLVES IT!

sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
train = pd.read_csv('train.csv').fillna('')
train.head()

# ================================== train ======================================== 

ct = train.shape[0]
input_ids = np.ones((ct,MAX_LEN),dtype='int32')
attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')
token_type_ids = np.zeros((ct,MAX_LEN),dtype='int32')
start_tokens = np.zeros((ct,MAX_LEN),dtype='int32')
end_tokens = np.zeros((ct,MAX_LEN),dtype='int32')

for k in range(train.shape[0]):
    
    # FIND OVERLAP
    text1 = " "+" ".join(train.loc[k,'text'].split())
    text2 = " ".join(train.loc[k,'selected_text'].split())
    idx = text1.find(text2)
    chars = np.zeros((len(text1)))
    chars[idx:idx+len(text2)]=1                             # character-position mask = 1 for text2 inside text1 
    if text1[idx-1]==' ': chars[idx-1] = 1                  # space before text2 is also included in mask
    enc = tokenizer.encode(text1) 
        
    # print(text1)
    # print(text2)
    # print(idx)
    # print(chars)
    # print(enc) # !!! THIS IS THE PROBLEM
    # print(enc.ids) # !!! THIS IS THE PROBLEM
    #                 # TODO: CHECK WHETHER THAT PROBLEM EXISTS TOO IN

    # TODO: Something doesn't work here!!
    # =====================================

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

        
    s_tok = sentiment_id[train.loc[k,'sentiment']]
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
    
    # print("============")
    # if(k==10): break
# Note down: the reason it "worked" with tf2.1 setup is that actually we had an old python with old pip,
# that initialized the tokenizer correctly so that it actually did its work. Taken care of now!


# ================== Test data preparation ==================
# Similar for test, as we did for train. Here, we only have token-sequences and sentiments
# (that means, since nothing is extracted, we'll just check whether the sentiment is correct?)
# Q: why exactly is the test-data only [(extracted/original ?)sequence, sentiment] enough for testing? 
#    i.e. Why is extraction not tested ?
test = pd.read_csv('test.csv').fillna('')

ct = test.shape[0]
input_ids_t = np.ones((ct,MAX_LEN),dtype='int32')
attention_mask_t = np.zeros((ct,MAX_LEN),dtype='int32')
token_type_ids_t = np.zeros((ct,MAX_LEN),dtype='int32') # NOTE: NOT used in RoBERTa, and probably don't help us either

for k in range(test.shape[0]):
        
    # INPUT_IDS
    text1 = " "+" ".join(test.loc[k,'text'].split())
    enc = tokenizer.encode(text1)                
    s_tok = sentiment_id[test.loc[k,'sentiment']]
    input_ids_t[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]
    attention_mask_t[k,:len(enc.ids)+5] = 1

# ================== Build Model ==================
def build_model():
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32) # (?, MAX_LEN)  Tokenized sentence "<s> {...tokens} </s></s> sentiment </s> ..(masked pads).."
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32) # (?, MAX_LEN)  Attention mask for non-padded part above
    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32) # (?, MAX_LEN)  unused part, Q: probably necessary for bert-base?

    config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')
    # well, I assume this is a roBERTa model, not a BERT model! Maybe to mislead chatgpt?
    bert_model = TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5',config=config)
    # bert_model = TFRobertaModel()

    # x[0] is a representation with hidden-size of the tokenized output (found substring representing sentiment)
    # In the following, it will be put through some layers 
    # x[0] is of (?, MAX_LEN, hidden_size=768). Layers turn it into (?, MAX_LEN)
    x = bert_model(ids,attention_mask=att,token_type_ids=tok)
    
    x1 = tf.keras.layers.Dropout(0.1)(x[0])         # (?, MAX_LEN, hidden_size)
    x1 = tf.keras.layers.Conv1D(1,1)(x1)            # (?, MAX_LEN, 1)
    x1 = tf.keras.layers.Flatten()(x1)              # (?, 1*MAX_LEN) = (?, MAX_LEN)
    x1 = tf.keras.layers.Activation('softmax')(x1)  # (?, MAX_LEN)

    x2 = tf.keras.layers.Dropout(0.1)(x[0])         
    x2 = tf.keras.layers.Conv1D(1,1)(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Activation('softmax')(x2)

    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])
    # note: the outputs are the one-hot predictions for start and end of of the indices within ids
    # that describe the sentiment

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer)


    return model

# ================== Loss metric ==================
# will be used to determine overlap of tokenized sentence's token-subsequence from
# prediction (later rextracted per predicted one-hot indices) and desired subsequence
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    if (len(a)==0) & (len(b)==0): return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))





model = build_model() # load HUK model
model.load_weights('weights_final.h5', by_name=True, skip_mismatch=False) # load HUK model

print("apparently success")

idXs = np.arange(0,100)

# # text1s = " "+" ".join(train.loc[idXs[0],'text'].split())
# print(input_ids[0])
# print(attention_mask[0])
# print(token_type_ids[0])
# # test.loc[idXs, 'text'])
# # print(test.loc[idXs, 'sentiment'])

# look mate, predict does loops. What about calling it normally
# is predict a call to model()?
# qq = model.predict(
#     [np.stack([input_ids[0]]),
#     np.stack([attention_mask[0]]),
#     np.stack([token_type_ids[0]])]
# )

a, b = model(    
    (np.stack([input_ids[0]]),
    np.stack([attention_mask[0]]),
    np.stack([token_type_ids[0]]))  )

print(a.shape)
print(b.shape)

# # # ================== train ==================
# jac = []; VER='v0'; DISPLAY=1 # USE display=1 FOR INTERACTIVE
# oof_start = np.zeros((input_ids.shape[0],MAX_LEN))
# oof_end = np.zeros((input_ids.shape[0],MAX_LEN))
# preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))
# preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))

# skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=777)
# for fold,(idxT,idxV) in enumerate(skf.split(input_ids, train.sentiment.values)):

#     print('#'*25)
#     print('### FOLD %i'%(fold+1))
#     print('#'*25)
    
#     # fresh model
#     K.clear_session()
#     model = build_model()
#     model.load_weights('weights_final.h5', by_name=True, skip_mismatch=False) # load HUK model

#     # # callback for saving
#     # sv = tf.keras.callbacks.ModelCheckpoint('%s-roberta-%i.h5'%(VER,fold), monitor='val_loss', verbose=1, save_best_only=True,
#     #     save_weights_only=True, mode='auto', save_freq='epoch')

#     # model = tf.keras.models.load_model('weights_final.h5') # load HUK model
#     # model.load_weights('weights_final.h5', by_name=True) # load HUK model
#     model.fit([input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]], [start_tokens[idxT,], end_tokens[idxT,]], 
#         epochs=3, batch_size=32  , verbose=DISPLAY, #callbacks=[sv],
#         validation_data=([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]], 
#         [start_tokens[idxV,], end_tokens[idxV,]]))
    
#     print('Loading model...')
#     model.load_weights('%s-roberta-%i.h5'%(VER,fold))
    
#     print('Predicting OOF...')
#     oof_start[idxV,],oof_end[idxV,] = model.predict([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]],verbose=DISPLAY)
    
#     print('Predicting Test...')
#     preds = model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)
#     preds_start += preds[0]/skf.n_splits    # this looks like it should have a preds_start[idV, ]
#     preds_end += preds[1]/skf.n_splits      # this looks like it should have a preds_end[idV, ]
    
#     # DISPLAY FOLD JACCARD
#     # this 
#     all = []
#     for k in idxV:
#         a = np.argmax(oof_start[k,])
#         b = np.argmax(oof_end[k,])
#         if a>b: 
#             st = train.loc[k,'text'] # IMPROVE CV/LB with better choice here
#         else:
#             text1 = " "+" ".join(train.loc[k,'text'].split())
#             enc = tokenizer.encode(text1)
#             st = tokenizer.decode(enc.ids[a-1:b])
#         all.append(jaccard(st,train.loc[k,'selected_text']))
#     jac.append(np.mean(all))
#     print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all))
#     print()