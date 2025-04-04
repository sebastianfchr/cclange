import pandas as pd, numpy as np, tensorflow as tf
import tensorflow.keras.backend as K
# from transformers import *
from sklearn.model_selection import StratifiedKFold
import tokenizers
from roberta import models, TokenEncoder, jaccard


MAX_LEN = 96


# Necessary facts for encodings: Tokenizer and sentiments
tokenizer = tokenizers.ByteLevelBPETokenizer.from_file('roberta/config/vocab-roberta-base.json', 'roberta/config/merges-roberta-base.txt', lowercase=True, add_prefix_space=True)
sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}

# Test and train dataset
test = pd.read_csv('./data/test.csv').fillna('')
train = pd.read_csv('./data/train.csv').fillna('')

# Use a 'TokenEncoder' to create the necessary ground-truth and test-data from 'test' and 'train' data 
# in the format used by RoBERTa
e = TokenEncoder(MAX_LEN, tokenizer)
input_ids, attention_mask, token_type_ids, start_tokens, end_tokens = e.prepare_encode_train(train)
input_ids_t, attention_mask_t, token_type_ids_t = e.prepare_encode_test(test)



# Kept the original training-loop unchanged:

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
    
    # fresh model for this part of the fold
    K.clear_session()
    model = models.build_model(MAX_LEN)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5))

    # Callback for saving. Saves every epoch, but overrides them within a fold. Only last epoch of fold remains
    sv = tf.keras.callbacks.ModelCheckpoint('%s-roberta-%i.h5'%(VER,fold), monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=True, mode='auto', save_freq='epoch')

    # model.load_weights('weights_final.h5', by_name=True, skip_mismatch=False) # load HUK model
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