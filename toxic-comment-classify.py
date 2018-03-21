import numpy as np
import pandas as pd
import math

from subprocess import check_output
print(check_output(["ls", "./input"]).decode("utf8"))

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from matplotlib import pyplot

max_features = 20000
maxlen = 100

# データセット読み込み
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

# サンプリング（割合100%）
train = train.sample(frac=1)

# 念のため、コメント部分のnullがある場合は特定の文字列に置換
list_sentences_train = train["comment_text"].fillna("ryok").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("ryok").values

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
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


model = get_model()
batch_size = 32 # バッチサイズ？なぜ
epochs = 2 #エポック数

# モデルの保存設定
file_path="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# 早期終了（最低ループ20回）
early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
callbacks_list = [checkpoint, early]

try:
    # 出力最後の20％を検証に利用
    history = model.fit(
        X_t, y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=callbacks_list
    )
except KeyboardInterrupt:
    pass

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# テストデータに対して予測実施
model.load_weights(file_path)
y_test = model.predict(X_te)

# 予測結果の出力
sample_submission = pd.read_csv("./input/sample_submission.csv")
sample_submission[list_classes] = y_test
sample_submission.to_csv("baseline.csv", index=False)