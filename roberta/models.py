
import tensorflow as tf
from transformers import RobertaConfig, TFRobertaModel

# VERSION 1: tf.keras.Model Style
# This is never used, but I'm just putting it in here in case it's considered better practice
import os 

class MyRobertaModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        # TODO: these paths come into init

    
        # load weights and config from within this package-dir
        package_dir = os.path.dirname(os.path.abspath(__file__))
        config = RobertaConfig.from_pretrained(os.path.join(package_dir, "config", "config-roberta-base.json"))
        self.roberta = TFRobertaModel.from_pretrained(os.path.join(package_dir, "config", "pretrained-roberta-base.h5"),config=config) 

        self.dropout_left = tf.keras.layers.Dropout(0.1)
        self.conv1d_left = tf.keras.layers.Conv1D(1,1)
        # right
        self.dropout_right = tf.keras.layers.Dropout(0.1)
        self.conv1d_right = tf.keras.layers.Conv1D(1,1)
        # these are transformations (no parameters or states): reusable
        self.flatten = tf.keras.layers.Flatten()
        self.softmax = tf.keras.layers.Activation('softmax')

    def call(self, id_att_tok, training=False):

        id, att, tok = id_att_tok[0], id_att_tok[1], id_att_tok[2]

        (x, _) = self.roberta(input_ids=id, attention_mask=att, token_type_ids=tok,  training=training, return_dict=False)
        # -> (?, MAX_LEN, hidden_size)

        xl = self.dropout_left(x, training=training)    # -> (?, MAX_LEN, hidden_size)
        xl = self.conv1d_left(xl)                       # -> (?, MAX_LEN, 1)
        xl = self.flatten(xl)                           # -> (?, MAX_LEN*1) = (?, MAX_LEN)
        xl = self.softmax(xl)                           # -> (?, MAX_LEN*1)

        xr = self.dropout_right(x, training=training)   # -> (?, MAX_LEN, hidden_size
        xr = self.conv1d_right(xr)                      # -> (?, MAX_LEN, 1
        xr = self.flatten(xr)                           # -> (?, MAX_LEN*1)
        xr = self.softmax(xr)                           # -> (?, MAX_LEN*1)

        return (xl, xr) # one-hot like encodings from sigmoid for left and right boundary of extracted token-sequence
     

# VERSION 2: TF API Style 
# Used this due to better inference in deployment, clearer layer-inspection and loading of weights

def build_model(max_len_tokens):

    ids = tf.keras.layers.Input((max_len_tokens,), dtype=tf.int32) # (?, MAX_LEN)  Tokenized sentence "<s> {...tokens} </s></s> sentiment </s> ..(masked pads).."
    att = tf.keras.layers.Input((max_len_tokens,), dtype=tf.int32) # (?, MAX_LEN)  Attention mask for non-padded part above
    tok = tf.keras.layers.Input((max_len_tokens,), dtype=tf.int32) # (?, MAX_LEN)  unused part, Q: probably necessary for bert-base?

    # load weights and config from within this package-dir. 
    package_dir = os.path.dirname(os.path.abspath(__file__))
    config = RobertaConfig.from_pretrained(os.path.join(package_dir, "config", "config-roberta-base.json"))
    bert_model = TFRobertaModel.from_pretrained(os.path.join(package_dir, "config", "pretrained-roberta-base.h5"), config=config)

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

    return model
