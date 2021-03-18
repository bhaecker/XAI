import sys
import pickle
import copy
import numpy as np

import skimage.segmentation
import skimage.io

import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

import sklearn.metrics
from sklearn.linear_model import LinearRegression

from model_zoo import create_cnn_model_RGB

###full training process for RGB images

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

percentage = 0.2
N = 100
image_id = 431
trainX = trainX[:N]
trainY = trainY[:N]
#make black background slighty lighter
trainX[:][:][trainX[:][:]<=0.01] = 0.01
trainX[:][:][:][trainX[:][:][:]<=0.01] = 0.01
images = np.stack((trainX[:][:][:], trainX[:][:][:], trainX[:][:][:]), axis=3).reshape((-1,28,28,3))

#make black background slighty lighter
testX[:][:][testX[:][:]<=0.01] = 0.01
testX[:][:][:][testX[:][:][:]<=0.01] = 0.01
images_test = np.stack((testX[:][:][:], testX[:][:][:], testX[:][:][:]), axis=3).reshape((-1,28,28,3))

model = create_cnn_model_RGB()
model.fit(images, trainY, epochs=1, batch_size=32, validation_data=(images_test, testY), verbose=1)
model.save('trained_RGB_model.h5')

#sys.exit()

####Explanation creation

sample_image=images_test[image_id]

plt.imshow(sample_image, interpolation='none')
plt.show()

superpixels = np.arange(28*28).reshape(28,28)
print(superpixels)
num_superpixels = np.unique(superpixels).shape[0]
skimage.io.imshow(skimage.segmentation.mark_boundaries(sample_image/2+0.5, superpixels))
plt.show()
#Generate perturbations
num_perturb = 2222
perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))

#Create function to apply perturbations to images
#function taken from Cristian Arteaga https://towardsdatascience.com/interpretable-machine-learning-for-image-classification-with-lime-ea947e82ca13
def perturb_image(img,perturbation,segments):
  active_pixels = np.where(perturbation == 1)[0]
  mask = np.zeros(segments.shape)
  for active in active_pixels:
      mask[segments == active] = 1
  perturbed_image = copy.deepcopy(img)
  perturbed_image = perturbed_image*mask[:,:,np.newaxis]
  return perturbed_image


trained_model = load_model("trained_RGB_model.h5")

predictions = []
for pert in perturbations:
	#skimage.io.imshow(perturb_image(sample_image/2+0.5,per,superpixels))
	#plt.show()
	perturbed_img = perturb_image(sample_image, pert, superpixels)
	pred = trained_model.predict(perturbed_img[np.newaxis, :, :, :])
	predictions.append(pred)
predictions = np.array(predictions)

no_perturbation = np.ones(num_superpixels)[np.newaxis,:]
distances = sklearn.metrics.pairwise_distances(perturbations,no_perturbation, metric='cosine').ravel()

#Transform distances to a value between 0 an 1 (weights) using a kernel function
kernel_width = 0.25
weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) #Kernel function

class_to_explain = np.argmax(testY[image_id])
simpler_model = LinearRegression()
simpler_model.fit(X=perturbations, y=predictions[:,:,class_to_explain], sample_weight=weights)
coeff = simpler_model.coef_[0]

coeff_image = coeff.reshape(28,28)

fig = plt.figure(figsize=(6, 3.2))

ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(coeff_image)
ax.set_aspect('equal')
plt.show()



