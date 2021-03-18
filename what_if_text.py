import sys
import re
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from model_zoo import create_LSTM_model_text_with_encoder
from explanation_methods import what_if_for_text

###what_if_for_text not possible - pls ignore this file
#full training process

movie_reviews = pd.read_csv("IMDB Dataset.csv")

movie_reviews.isnull().values.any()

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence
X = []
sentences = list(movie_reviews['review'])
for sen in sentences:
    X.append(preprocess_text(sen))

y = movie_reviews['sentiment']

y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train+X_test)
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
pickle.dump(tokenizer,open('tokenizer.pickle', 'wb'))

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1

maxlen = 600

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

np.save("X_train",X_train)
np.save("X_test",X_test)
np.save("y_train",X_train)
np.save("y_test",y_test)

embeddings_dictionary = dict()
glove_file = open('glove.6B.200d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

embedding_matrix = np.zeros((vocab_size, 200))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

np.save("embedding_matrix_200d",embedding_matrix)

model = create_LSTM_model_text_with_encoder(vocab_size,maxlen,embedding_matrix)
model.summary()

history = model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=1, validation_split=0.2)

model.save("trained_LSTM_auto_text_model.h5")

#sys.exit()

####load everything and do what_if_for_text

maxlen = 600

X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

with open("tokenizer.pickle", 'rb') as pickle_file:
    tokenizer = pickle.load(pickle_file)

reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

text_id = 2

sample_text = X_test[text_id]

class_to_explain = y_test[text_id]
target = (class_to_explain +1) % 2

embedding_matrix = np.load("embedding_matrix_200d.npy")

model = load_model("trained_LSTM_auto_text_model.h5")
model.summary()

print('everything loaded, start counter factual creation')

print(what_if_for_text(model,sample_text,class_to_explain,reverse_word_map,0.8))








