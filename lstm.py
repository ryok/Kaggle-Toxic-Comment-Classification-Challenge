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
import matplotlib.pyplot as plt

max_features = 20000
maxlen = 200

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
    embed_size = 128 # embeddingのoutputサイズ
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = LSTM(60, return_sequences=True, name='lstm_layer')(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
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
batch_size = 32 # バッチサイズ
epochs = 10 # エポック数

# モデルの保存設定
file_path="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# 早期終了
early = EarlyStopping(monitor="val_loss", mode="min", patience=3)

#X_tra, X_val, y_tra, y_val = train_test_split(X_t, y, train_size=0.95, random_state=1)
X_tra, X_test, y_tra, y_test = train_test_split(X_t, y, train_size=0.95, random_state=1)
#RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
callbacks_list = [checkpoint, early]

try:
    # 交差検定
    history = model.fit(
        X_tra, y_tra,
        batch_size=batch_size,
        epochs=epochs,
        #validation_data=(X_val, y_val),
        validation_split=0.2,
        callbacks=callbacks_list
    )
except KeyboardInterrupt:
    pass

score = model.evaluate(X_test, y_test,
                       verbose=0
                       )

print('Test score:', score[0])
print('Test accuracy:', score[1])

# ----------------------------------------------
# Some plots
# ----------------------------------------------
fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

# loss
def plot_history_loss(fit):
    # Plot the loss in the history
    axL.plot(fit.history['loss'],label="loss for training")
    axL.plot(fit.history['val_loss'],label="loss for validation")
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')

# acc
def plot_history_acc(fit):
    # Plot the loss in the history
    axR.plot(fit.history['acc'],label="acc for training")
    axR.plot(fit.history['val_acc'],label="acc for validation")
    axR.set_title('model accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.legend(loc='upper right')

plot_history_loss(history)
plot_history_acc(history)
fig.savefig('./pattern1.png')
plt.close()

# テストデータに対して予測実施
#model.load_weights(file_path)
#y_test = model.predict(X_te)

# 予測結果の出力
#sample_submission = pd.read_csv("./input/sample_submission.csv")
#sample_submission[list_classes] = y_test
#sample_submission.to_csv("baseline.csv", index=False)