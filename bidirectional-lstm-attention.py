import numpy as np
import pandas as pd
import math

from subprocess import check_output
#print(check_output(["ls", "./input"]).decode("utf8"))

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.callbacks import Callback
from matplotlib import pyplot

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


max_features = 20000
maxlen = 100

# データセット読み込み
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

# サンプリング（割合100%）
train = train.sample(frac=1)

# 念のため、コメント部分のnullがある場合は特定の文字列に置換
list_sentences_train = train["comment_text"].fillna("unknown").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("unknown").values

# テストデータのサイズ（量）
vocab_size = len(list_sentences_test)

# Tokenizerによる文字列の数値化
tokenizer = text.Tokenizer(num_words=max_features) # データセット中の頻度上位num_wordsの単語に制限
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

# maxlenにpaddingし、長さを揃える
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

def get_model():
    embed_size = 128 #embeddingのoutputサイズ
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    _, *encoder_states = LSTM(embed_size, return_state=True)(x)

    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = Attention(maxlen)(x)
    #x = GlobalMaxPool1D()(x)
    #x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))


model = get_model()
batch_size = 32 # バッチサイズ？なぜ
epochs = 50 #エポック数

# モデルの保存設定
file_path="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# 早期終了（最低ループ20回）
early = EarlyStopping(monitor="val_loss", mode="min", patience=20)

X_tra, X_val, y_tra, y_val = train_test_split(X_t, y, train_size=0.95, random_state=1)
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
callbacks_list = [checkpoint, early, RocAuc]

try:
    # 交差検定
    history = model.fit(
        X_t, y,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        #validation_split=0.2,
        callbacks=callbacks_list
    )
except KeyboardInterrupt:
    pass

print(history.history)


# lossのプロット
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# lossのプロット
pyplot.plot(history.history['acc'], label='train')
pyplot.plot(history.history['val_acc'], label='test')
pyplot.legend()
pyplot.show()

# テストデータに対して予測実施
model.load_weights(file_path)
y_test = model.predict(X_te)

# 予測結果の出力
sample_submission = pd.read_csv("./input/sample_submission.csv")
sample_submission[list_classes] = y_test
sample_submission.to_csv("baseline.csv", index=False)