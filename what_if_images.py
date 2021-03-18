import sys
import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

from model_zoo import create_cnn_model_with_encoder
from explanation_methods import what_if_for_mnsit



####full training process and counterfactual creation

def load_dataset(pixels):
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# reshape dataset to have a single channel

	trainX = trainX.reshape((trainX.shape[0], pixels, pixels, 1))
	testX = testX.reshape((testX.shape[0], pixels, pixels, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

pixels = 28

trainX, trainY, testX, testY = load_dataset(pixels)
trainX, testX = prep_pixels(trainX,testX)

trained_model = create_cnn_model_with_encoder()
trained_model.fit(trainX, trainY, epochs=1, batch_size=32, validation_data=(testX, testY), verbose=1)
trained_model.save("trained_base_model.h5")

sample = trainX[11:12]
plt.imshow(sample.reshape((28, 28)), cmap='gray', interpolation='none')
plt.show()

trained_model = load_model("trained_base_model.h5")

new_sample = what_if_for_mnsit(trained_model, sample, 7, 100, 0.75)

plt.imshow(new_sample.reshape((28, 28)), cmap='gray', interpolation='none')
plt.show()

#sys.exit()

fig = plt.figure()
for i in range(9):
	trained_model = load_model("trained_base_model.h5")
	new_sample = what_if_for_mnsit(trained_model, sample, i, 100, 0.75)
	plt.subplot(3,3,i+1)
	plt.tight_layout()
	plt.imshow(new_sample.reshape((28, 28)), cmap='gray', interpolation='none')
	plt.title("Digit: {}".format(i))
	plt.xticks([])
	plt.yticks([])
plt.show()