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


train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
train = train.sample(frac=1)

list_sentences_train = train["comment_text"].fillna("CVxTz").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("CVxTz").values

vocab_size = len(list_sentences_test)

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

def get_model():
    embed_size = 128
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


def generator(data_x, data_y, batch_size=32):
    n_batches = math.ceil(len(data_x) / batch_size)

    while True:
        for i in range(n_batches):
            start = i * batch_size
            end = (i + 1) * batch_size

            data_x_mb = data_x[start:end]
            data_y_mb = data_y[start:end]

            data_x_mb = np.array(data_x_mb).astype('float32') / 255.
            data_y_mb = pad_sequences(data_y_mb, dtype='int32', padding='post', value=w2i['<pad>'])
            data_y_mb_oh = np.array([np_utils.to_categorical(datum_y, vocab_size) for datum_y in data_y_mb[:, 1:]])

            yield [data_x_mb, data_y_mb], data_y_mb_oh

model = get_model()
batch_size = 32
epochs = 2


file_path="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early = EarlyStopping(monitor="val_loss", mode="min", patience=20)


callbacks_list = [checkpoint, early] #early
try:
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

# make a prediction
model.load_weights(file_path)
y_test = model.predict(X_te)

sample_submission = pd.read_csv("./input/sample_submission.csv")
sample_submission[list_classes] = y_test
sample_submission.to_csv("baseline.csv", index=False)