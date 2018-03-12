import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import Callback

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

path = '../input/'
comp = 'jigsaw-toxic-comment-classification-challenge/'
EMBEDDING_FILE=path + 'glove.6B.50d.txt'
TRAIN_DATA_FILE= path+'train.csv'
TEST_DATA_FILE= path +'test.csv'


embed_size = 50 # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a comment to use


train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)

list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values


print('start preprocessing data')
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)


def get_coefs(word,*arr):
	return word, np.asarray(arr, dtype='float32')




class RocAucEvaluation(Callback):
	def __init__(self, validation_data=(), interval=1):
		super(Callback, self).__init__()

		self.interval = interval
		self.X_val, self.y_val = validation_data

	def on_epoch_end(self, epoch, logs={}):
		if epoch % self.interval == 0:
			y_pred = self.model.predict(self.X_val, verbose=0)
			score = roc_auc_score(self.y_val, y_pred)
			print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch + 1, score))




embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))


all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std


print('start embedding')
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
	if i >= max_features: continue
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None: embedding_matrix[i] = embedding_vector



print('start training')
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_tra, X_val, y_tra, y_val = train_test_split(X_t, y, test_size=0.1)

RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1) 

callbacks = [RocAuc]


print('start fitting')
#model.fit(X_t, y, batch_size=32, epochs=2, validation_split=0.1, callbacks=callbacks);
history = model.fit(X_tra, y_tra, batch_size=32, epochs=2, validation_data=(X_val, y_val),
					 callbacks=callbacks, verbose=2, shuffle=True)


y_pred = model.predict(self.X_val, verbose=0)
score = roc_auc_score(y_val, y_pred)
print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch + 1, score))

#summarize history for Accuracy
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('Acc')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
fig_acc.savefig(base_path_output + "model_accuracy.png")

# summarize history for loss
fig_loss = plt.figure(figsize=(10, 10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.show()
fig_loss.savefig(base_path_output + "model_loss.png")

y_test = model.predict([X_te], batch_size=1024, verbose=1)
sample_submission = pd.read_csv(path + 'sample_submission.csv')
sample_submission[list_classes] = y_test
sample_submission.to_csv('submission.csv', index=False)





