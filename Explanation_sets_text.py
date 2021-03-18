import sys
import pickle
import numpy as np

from tensorflow.keras.models import load_model

from explanation_methods import make_explantation_from_distances_for_text


####load everything (full training process in LIME_text.py)

maxlen = 600

X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print(y_test.shape)

with open("tokenizer.pickle", 'rb') as pickle_file:
    tokenizer = pickle.load(pickle_file)

reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

text_id = 1218#3122#STARWARS: 7267

sample_text = X_test[text_id]

embedding_matrix = np.load("embedding_matrix_200d.npy")

model = load_model("trained_LSTM_text_model_2epochs.h5")

print('everything loaded, start make_explantation_from_distances_for_text')
print(make_explantation_from_distances_for_text(X_train[:2000],model,sample_text,10,reverse_word_map))



