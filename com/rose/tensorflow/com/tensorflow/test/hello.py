# http://www.fuzihao.org/blog/2016/02/29/Predict-Time-Sequence-with-LSTM/
# periodically LSTM has timesequence...
# http://qiita.com/t_Signull/items/21b82be280b46f467d1b
# succ after 5040 is LSTM!
# http://kivantium.hateblo.jp/entry/2016/01/31/222050
# chainner case 10000
import itertools
import numpy as np

sentences = '''
sam is red
hannah not red
hannah is green
bob is green
bob not red
sam not green
sarah is red
sarah not green'''.strip().split('\n')
is_green = np.asarray([[0, 1, 1, 1, 1, 0, 0, 0]], dtype='int32').T

lemma = lambda x: x.strip().lower().split(' ')
sentences_lemmatized = [lemma(sentence) for sentence in sentences]
words = set(itertools.chain(*sentences_lemmatized))
# set(['boy', 'fed', 'ate', 'cat', 'kicked', 'hat'])

# dictionaries for converting words to integers and vice versa
word2idx = dict((v, i) for i, v in enumerate(words))
idx2word = list(words)

# convert the sentences a numpy array
to_idx = lambda x: [word2idx[word] for word in x]
sentences_idx = [to_idx(sentence) for sentence in sentences_lemmatized]
sentences_array = np.asarray(sentences_idx, dtype='int32')

# parameters for the model
sentence_maxlen = 3
n_words = len(words)
n_embed_dims = 3

# put together a model to predict
from keras.layers import Input, Embedding, merge, Flatten, recurrent
from keras.models import Model
from theano.tensor import unbroadcast
#from keras.layers.recurrent import LSTM
RNN = recurrent.LSTM
#RNN = LSTM

print(words)
print(sentences_lemmatized)
print(sentences_array)

#{'red', 'green', 'bob', 'not', 'sam', 'hannah', 'sarah', 'is'}
#[['sam', 'is', 'red'], ['hannah', 'not', 'red'], ['hannah', 'is', 'green'], ['bob', 'is', 'green'], ['bob', 'not', 'red'], ['sam', 'not', 'green'], ['sarah', 'is', 'red'], ['sarah', 'not', 'green']]
#[[4 7 0]
# [5 3 0] 1 hannah
# [5 7 1] 1
# [2 7 1] 1 bob
# [2 3 0] 1
# [4 3 1]
# [6 7 0]
# [6 3 1]]

input_sentence = Input(shape=(sentence_maxlen,), dtype='int32')
input_embedding =Embedding(n_words, n_embed_dims)(input_sentence)
# https://github.com/Theano/Theano/issues/1772
input_embedding = unbroadcast(input_embedding,0)
#input_embedding = unbroadcast(input_embedding,1)
color_prediction = RNN(1)(input_embedding)

predict_green = Model(input=[input_sentence], output=[color_prediction])
#http://stackoverflow.com/questions/37523882/keras-wrappers-for-scikit-learn-auc-scorer-is-not-working
#accuracy
#predict_green.compile(optimizer='sgd',metrics=["accuracy"], loss='binary_crossentropy')
predict_green.compile(optimizer='sgd', loss='binary_crossentropy')
predict_green.trainable = True
# fit the model to predict what color each person is
hist = predict_green.fit([sentences_array], [is_green], nb_epoch=5500, verbose=1,validation_split=0.2)
print(hist.history)
print(predict_green.layers[1].W.get_value())
embeddings = predict_green.layers[1].W.get_value()
# dump,layer
# http://aidiary.hatenablog.com/entry/20150626/1435329581
# print out the embedding vector associated with each word
# https://github.com/fchollet/keras/issues/3075
# Is there any way to update only a subset of model parameters when using model.fit() procedure? For example, update only the weights corresponding to model.layers[1].W
for i in range(n_words):
	print('{}: {}'.format(idx2word[i], embeddings[i]))
