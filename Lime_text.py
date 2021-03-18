import sys
import re
import pickle
import numpy as np
import pandas as pd

import sklearn.metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
from matplotlib import colors
from highlight_text import ax_text

from model_zoo import create_LSTM_model_text


class MidpointNormalize(colors.Normalize):
    """Normalise the colorbar."""
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

##full training process

movie_reviews = pd.read_csv("IMDB Dataset.csv")

#preprocessing from Jason Brownlee https://machinelearningmastery.com/prepare-movie-review-data-sentiment-analysis/
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

model = create_LSTM_model_text(vocab_size,maxlen,embedding_matrix)
model.summary()
history = model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=1, validation_split=0.2)
model.save("trained_LSTM_text_model_5epochs.h5")

#sys.exit()

####load everything and do LIME for text

maxlen = 600

X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

with open("tokenizer.pickle", 'rb') as pickle_file:
    tokenizer = pickle.load(pickle_file)

reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

text_id = 3441#STARWARS: 7267

sample_text = X_test[text_id]
class_to_explain = y_test[text_id]

embedding_matrix = np.load("embedding_matrix_200d.npy")

model = load_model("trained_LSTM_text_model.h5")

print('everything loaded, start pertubations')

num_perturb = 555
perturbations = np.random.binomial(1, 0.5, size=(num_perturb, maxlen))

def perturb_text(text,perturbation):
    return np.where(perturbation == 1, text, perturbation)

trained_model = load_model("trained_LSTM_text_model.h5")

predictions = []
for count,pert in enumerate(perturbations):
    perturbed_text = perturb_text(sample_text, pert)[np.newaxis,:]
    pred = trained_model.predict(perturbed_text)[0]
    predictions.append(pred)
    print(str(count/num_perturb*100)+" %")

predictions = np.array(predictions)

original_text = np.ones(maxlen)

distances = sklearn.metrics.pairwise_distances(perturbations,original_text.reshape(1, -1), metric='cosine').ravel()

#Transform distances to a value between 0 an 1 (weights) using a kernel function
kernel_width = 0.25
weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) #Kernel function

simpler_model = LinearRegression()
simpler_model.fit(X=perturbations, y=predictions, sample_weight=weights)
coeff = simpler_model.coef_#[0]

#Use coefficients
num_top_features = maxlen
coeff_sorted = np.sort(coeff)[0]
top_features = np.argsort(coeff)[0]

readable_sample_text = [reverse_word_map.get(idx) for idx in sample_text]
print(readable_sample_text)
if class_to_explain == 1:
    print('positive')
    string = "-------------------------positive-------------------------\n"
else:
    print('negative')
    string = "-------------------------negative-------------------------\n"

for coeff, token in zip(coeff_sorted,top_features):
    print(coeff, reverse_word_map.get(token))

top_words = [readable_sample_text[idx] for idx in top_features]

minima = min(coeff_sorted)#-1  # min(weights)
maxima = max(coeff_sorted)#1  # max(weights)
norm = MidpointNormalize(minima, maxima, 0.)
weights = norm(coeff_sorted)
color = plt.cm.coolwarm(weights)

colors = []
counter = 0
for sentence_word in readable_sample_text:
    if sentence_word is None:
        break
    if counter >= 55:
        string += "\n"
        counter = 0
    if sentence_word in top_words:
        idx = top_words.index(sentence_word)
        print(idx)
        print(coeff_sorted[idx])
        colors.append(color[idx])
        string += " "+"<"+sentence_word+">"
    else:
        string += " " +  sentence_word
    counter += len(sentence_word) + 1

fig,ax = plt.subplots(1,figsize=(6,6))
ax_text(x = 0, y = 0,
        s = string, color = 'k', highlight_colors = colors)
plt.axis('off')
plt.show()







