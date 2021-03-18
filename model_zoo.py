import sys
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, InputLayer, Input, Concatenate, LSTM, Layer, Reshape, Embedding
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.optimizers import SGD

tf.random.set_seed(42)

def create_model(number_features,number_classes,number_additional_hidden_layers,number_neurons_hidden_layer):
    '''
    create a small vanilla keras model, which is always initialized with the same weights
    '''
    print('create a NN model with '+str(1+number_additional_hidden_layers)+ ' hidden layer(s) with '+str(number_neurons_hidden_layer)+ ' neurons each')
    input_layer = Input(number_features,)
    input_layer_2 = Dense(number_features, trainable=False, activation=None)(input_layer)#,
                          #kernel_initializer=initializers.Constant(value=1/number_features),
                        #bias_initializer=initializers.Zeros())(input_layer)
    #initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed=42)
    x = Dense(number_neurons_hidden_layer,trainable=True, activation='relu')(input_layer_2)
    for i in range(number_additional_hidden_layers):
        x = Dense(number_neurons_hidden_layer,trainable=True, activation='relu')(x)
    predictions = Dense(number_classes,trainable=True, activation='softmax')(x)
    model = Model(inputs=input_layer,outputs=predictions)

    weights = [np.identity(number_features), np.zeros((number_features,))]
    model.layers[1].set_weights(weights)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return(model)


def train_model(model,epochs,Xtrain,ytrain):
    print('train for '+str(epochs)+' epochs')
    es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=5)
    #mc = ModelCheckpoint('best_model.h5', monitor='loss', mode='min', verbose=1, save_best_only=False)
    model.fit(Xtrain,ytrain,epochs=epochs,batch_size=128,verbose=1,validation_split=0.2,callbacks=[es])#,mc])

def create_cnn_model():
    model = Sequential()
    # actual model
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))

    opt = SGD(lr=0.0001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def create_cnn_model_RGB():
    '''CNN for RGB images'''
    model = Sequential()
    #actual model
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))

    opt = SGD(lr=0.0001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def create_cnn_model_with_encoder():
    model = Sequential()
    # encoder decoder
    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(28*28, trainable=False, activation='relu'))
    model.add(Reshape((28, 28, 1),input_shape=(28*28,)))
    # actual model
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))#, input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # initialize encoder decoder as identity function
    weights = [np.identity(28 * 28), np.zeros((28 * 28,))]
    model.layers[1].set_weights(weights)

    opt = SGD(lr=0.0001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def create_cnn_model_RGB_with_encoder():
    '''CNN for RGB images'''
    model = Sequential()
    #encoder decoder
    model.add(Flatten(input_shape=(28, 28, 3)))
    #model.add(Input)
    model.add(Dense(28*28, trainable=False, activation='relu'))
    model.add(Reshape((28, 28, 1),input_shape=(28*28,)))
    #actual model
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))#, input_shape=(28, 28, 3)
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # initialize encoder decoder as identity function
    weights = [np.identity(28 * 28), np.zeros((28 * 28,))]
    model.layers[1].set_weights(weights)

    opt = SGD(lr=0.0001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def create_LSTM_model_text(vocab_size,maxlen,embedding_matrix):
    model = Sequential()
    embedding_layer = Embedding(vocab_size, 200, weights=[embedding_matrix], input_length=maxlen, trainable=False)
    model.add(embedding_layer)
    model.add(LSTM(128, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
    model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5, return_sequences=False))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    return model


#does not work since Embeddig layer is not differentiable
def create_LSTM_model_text_with_encoder(vocab_size,maxlen,embedding_matrix):
    model = Sequential()
    #model.add(Flatten(input_shape=(maxlen,)))
    model.add(Input(shape=(maxlen,)))
    model.add(Dense(maxlen, trainable=False, activation='relu'))
    # model.add(Dense(28*28, trainable=False, activation='relu'))
    # model.add(Reshape((28, 28, 1),input_shape=(28*28,)))
    #model.add(Lambda(lambda x: K.round(x)))
    embedding_layer = Embedding(vocab_size, 200, weights=[embedding_matrix], input_length=maxlen, trainable=False)
    model.add(embedding_layer)
    model.add(LSTM(128, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
    model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5, return_sequences=False))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))

    weights = [np.identity(maxlen), np.zeros((maxlen,))]
    model.layers[0].set_weights(weights)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

    return model