import sys
import numpy as np
from numpy import linalg as LA
from scipy.spatial.distance import cdist,cosine
from tensorflow.keras.models import Model, load_model

from tensorflow.keras.optimizers import SGD
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt



def get_all_layer_outputs(basic_model,training_data):
    '''
    returns a list of length number of hidden layers + 1 (for output layer),
    where every element is an array of the activation for each training sample for the respective layer
    '''
    intermediate_model = Model(inputs=basic_model.layers[0].input,
                              outputs=[l.output for l in basic_model.layers[1:]])
    activation_set = intermediate_model.predict(training_data)
    #for activation in activation_set:
        #print(np.shape(activation))
        #print(activation)
    #print(len(activation_set))
    return activation_set


####for number data sets:

def make_explantation_from_distances(training_data,model,sample,number_explanations,linear_importance):
    '''
    calculates the euclidean distance between every activation of sample and training data
    and returns number_explanations closest training samples
    linear_importance is float number between 0 and 1 and weights the layer output with liner weights starting from linear_importance to 1
    '''

    for layer in model.layers:
        print(layer.__class__.__name__)

    weight = [1 if layer.__class__.__name__ == "Dense" else 0 for layer in model.layers]
    print(weight)

    layer_outputs_training = get_all_layer_outputs(model,training_data)
    layer_outputs_sample =  get_all_layer_outputs(model,sample)

    number_training_samples = np.shape(training_data)[0]
    number_layers = len(layer_outputs_sample)
    distance_array = np.empty(shape=(number_training_samples,number_layers))

    for idx, (layer_output_training, layer_output_sample) in enumerate(zip(layer_outputs_training,layer_outputs_sample)):
        distances = cdist(layer_output_training,layer_output_sample,metric='euclidean')
        distance_array[:,idx] = np.transpose(distances)
    weights = np.linspace(start=linear_importance, stop=1, num=number_layers)
    sum_weighted_distance_array = np.dot(distance_array,weights)

    #find 'number_explanations' smallest values !!!Attention, does not return them sorted
    idx_winner = np.argpartition(sum_weighted_distance_array, number_explanations)[:number_explanations]

    explantation_samples = training_data[idx_winner]
    return_message = "When the model 'saw' \n" \
                     + str(sample[0]) \
                     + "\n it was 'thinking' the same as when it 'saw' \n" \
                     + str(explantation_samples)

    return return_message

def what_if(trained_model,number_additional_layers,sample,class_id,epochs,confidence):

    trained_model.layers[1].trainable = True

    for idx in range(2,4 + number_additional_layers):
        print(idx)
        trained_model.layers[idx].trainable = False

    trained_model.compile(loss='sparse_categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
    trained_model.save('trained_toy_model.h5')

    sample = np.asarray([sample])

    target = np.asarray([class_id])

    for epoch in range(epochs):
        trained_model.fit(sample, target, epochs=1, batch_size=1, verbose=2)
        new_sample = get_all_layer_outputs(trained_model, sample)[0]
        print("new_sample")
        print(new_sample)
        print("prediction of new sample")#TODO: PREDICT WITH OLD MODEL !!
        base_model = load_model("trained_toy_model.h5")
        pred = base_model.predict(new_sample)
        print(pred)
        pred_class = np.argmax(pred[0])
        print(pred_class)

        if pred_class == class_id and pred[0][class_id] >= confidence:
            return " In order to be classified "+str(sample[0])+" in class "+ str(pred_class) +" with at least "+str(confidence*100)+"% confidence,\n " \
            "the sample would need to look like "+str(new_sample[0])+ ", which "+str(["really" if np.argmax(new_sample[0]) != class_id else "indeed"][0]) +" belongs to class "+ str(np.argmax(new_sample[0]))+"."

####for text classification

def what_if_for_text(trained_model,sample,class_id,reverse_word_map,confidence):
    """
    does not work since Embedding layer is not trainable
    """

    print(trained_model.summary())

    for idx in range(0,7):
        trained_model.layers[idx].trainable = False

    trained_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    print("nothing trainable")
    print(trained_model.summary())

    trained_model.layers[0].trainable = True
    print("encoder trainable")

    trained_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    print(trained_model.summary())
    sample = sample.reshape(1, -1)

    output_list = get_all_layer_outputs(trained_model,sample)

    new_text = [reverse_word_map.get(idx) for idx in output_list[0][0]]

    #just for entering while loop
    pred_class = class_id + 1
    pred = [[0]]
    target = np.asarray([class_id]).reshape(1,-1)
    while not (pred_class == class_id and pred[0][class_id] >= confidence):
        trained_model.fit(sample, target, epochs=1, batch_size=1, verbose=1)
        new_sample = get_all_layer_outputs(trained_model, sample)[1]
        print("predicted class of new sample")
        base_model = load_model("trained_LSTM_auto_text_model.h5")
        pred = base_model.predict(new_sample)
        pred_class = np.argmax(pred[0])
        print(pred_class)
        print(pred[0][pred_class])
        del base_model

    return new_sample

def make_explantation_from_distances_for_text(training_data,model,sample,number_explanations,reverse_word_map,linear_importance):
    '''
    calculates the euclidean distance between every activation of sample and training data
    and returns number_explanations closest training samples
    linear_importance is float number between 0 and 1 and weights the layer output with liner weights starting from linear_importance to 1
    '''

    for layer in model.layers:
        print(layer.__class__.__name__)

    sample = sample.reshape(1, -1)

    layer_outputs_training = get_all_layer_outputs(model,training_data)
    layer_outputs_sample =  get_all_layer_outputs(model,sample)

    for i,layer in enumerate(layer_outputs_training):
        layer_outputs_training[i] = layer.reshape(layer.shape[0],-1)
    for i,layer in enumerate(layer_outputs_sample):
        layer_outputs_sample[i] = layer.reshape(layer.shape[0], -1)

    number_training_samples = np.shape(training_data)[0]
    number_layers = len(layer_outputs_sample)
    distance_array = np.empty(shape=(number_training_samples,number_layers))

    for idx, (layer_output_training, layer_output_sample) in enumerate(zip(layer_outputs_training,layer_outputs_sample)):
        for sample_idx, single_layer_output_training in enumerate(layer_output_training):
            distances = cosine(single_layer_output_training,layer_output_sample)
            distance_array[sample_idx,idx] = np.transpose(distances)

    #weights = [1 if layer.__class__.__name__ == "Dense" else 0 for layer in model.layers][1:]
    weights = np.linspace(start=linear_importance, stop=1, num=number_layers)

    sum_weighted_distance_array = np.dot(distance_array,weights)

    #find 'number_explanations' smallest values !!!Attention, does not return them sorted
    idx_winner = np.argpartition(sum_weighted_distance_array, number_explanations)[:number_explanations]

    explantation_samples = training_data[idx_winner]
    print('When the model "saw"')
    list_string  = [reverse_word_map.get(idx) for idx in sample[0]]
    sentence = " ".join(list(filter(None, list_string)))
    print(sentence)

    print('the neuron activity was similar to when it "saw" the following training samples:')
    for sample in explantation_samples:
        list_string = [reverse_word_map.get(idx) for idx in sample]
        sentence = " ".join(list(filter(None, list_string)))
        print(sentence)

    return 'done'

###for image classification

def what_if_for_mnsit(trained_model, sample, class_id, confidence):
    '''
    trained_model must have an encoder decoder structure in the first three layers
    :returns alternated sample which is classified into class_id with confidence*100 %
    '''

    for idx in range(2, 11):
        trained_model.layers[idx].trainable = False

    opt = SGD(lr=0.0001, momentum=0.9)
    trained_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    trained_model.layers[1].trainable = True

    opt = SGD(lr=0.0001, momentum=0.9)
    trained_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    print(trained_model.summary())

    target = to_categorical(class_id, num_classes=10).reshape(1, 10)

    # just for entering while loop
    pred_class = class_id + 1
    pred = [[0]]
    while not (pred_class == class_id and pred[0][class_id] >= confidence):
        trained_model.fit(sample, target, epochs=1, batch_size=1, verbose=1)
        new_sample = get_all_layer_outputs(trained_model, sample)[1]
        print("predicted class of new sample")
        base_model = load_model("trained_base_model.h5")
        pred = base_model.predict(new_sample)
        pred_class = np.argmax(pred[0])
        print(pred_class)
        print(pred[0][pred_class])
        del base_model
    return new_sample



def make_explantation_from_distances_for_mnist(training_data,model,sample,number_explanations,linear_importance):
    '''
    calculates the euclidean distance between every activation of sample and training data
    and returns number_explanations closest training samples
    linear_importance is float number between 0 and 1 and weights the layer output with liner weights starting from linear_importance to 1
    '''

    for layer in model.layers:
        print(layer.__class__.__name__)

    layer_outputs_training = get_all_layer_outputs(model,training_data)
    layer_outputs_sample =  get_all_layer_outputs(model,sample)

    for i,layer in enumerate(layer_outputs_training):
        layer_outputs_training[i] = layer.reshape(layer.shape[0],-1)
    for i,layer in enumerate(layer_outputs_sample):
        layer_outputs_sample[i] = layer.reshape(layer.shape[0], -1)


    number_training_samples = np.shape(training_data)[0]
    number_layers = len(layer_outputs_sample)
    distance_array = np.empty(shape=(number_training_samples,number_layers))

    for idx, (layer_output_training, layer_output_sample) in enumerate(zip(layer_outputs_training,layer_outputs_sample)):
        for sample_idx, single_layer_output_training in enumerate(layer_output_training):
            distances = cosine(single_layer_output_training,layer_output_sample)
            distance_array[sample_idx,idx] = np.transpose(distances)

    #weights = [1 if layer.__class__.__name__ == "Conv2D" else 1 for layer in model.layers][1:]
    weights = np.linspace(start=linear_importance, stop=1, num=number_layers)
    sum_weighted_distance_array = np.dot(distance_array,weights)

    #find 'number_explanations' smallest values !!!Attention, does not return them sorted
    idx_winner = np.argpartition(sum_weighted_distance_array, number_explanations)[:number_explanations]

    explantation_samples = training_data[idx_winner]
    print('when the model saw')
    plt.imshow(sample.reshape((28,28)), cmap='gray', interpolation='none')
    plt.show()
    print('it was similar to')
    for sample in explantation_samples:
        plt.imshow(sample.reshape((28,28)), cmap='gray', interpolation='none')
        plt.show()

    return 'done'


def create_explanation_sets_for_mnist(trained_model,training_data,training_data_y,sample_id,number_explantation_sets):
    '''
    get unsupervised clusters for activations training samples and use them as explanation sets
    '''
    number_samples = training_data.shape[0]
    for layer in trained_model.layers:
        print(layer.__class__.__name__)

    layer_outputs_training = get_all_layer_outputs(trained_model, training_data)

    #weights = np.linspace(start=linear_importance, stop=1, num=number_layers)
    weights = [1 if layer.__class__.__name__ == "Conv2D" else 0 for layer in trained_model.layers][1:]
    weights[0]=0#get rid of Autoencoder Dense layer


    for i, layer in enumerate(layer_outputs_training):
        layer_outputs_training[i] = layer.reshape(layer.shape[0], -1)

    #mapping to 2 or 3d space of flattend activations
    vec_space_dimension = 0
    for layer_output, weight in zip(layer_outputs_training,weights):
        if weight != 0:
            vec_space_dimension += layer_output.shape[1]

    vector_space = np.empty(shape=(number_samples,vec_space_dimension))

    for sample_idx in range(number_samples):
        layer_dimension = 0
        for layer_idx,layer in enumerate(layer_outputs_training):
            if weights[layer_idx] != 0:
                current_layer_dimension = layer.shape[1]
                vector_space[sample_idx][layer_dimension:current_layer_dimension+layer_dimension] = layer[sample_idx][:]
                layer_dimension += current_layer_dimension

    model = TSNE(n_components = 2,learning_rate=100)
    transformed = model.fit_transform(vector_space)
    x_axis = transformed[:, 0]
    y_axis = transformed[:, 1]
    #z_axis = transformed[:, 2]

    data_y = [None] * number_samples
    for idx, single_training_data_y in enumerate(training_data_y):
        data_y[idx] = np.argmax(single_training_data_y)
    #from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_axis, y_axis, c = data_y)
    plt.show()

    #unsupervised clustering with fixed number of clusters of activations
    model = KMeans(n_clusters=number_explantation_sets)
    model.fit(vector_space)
    pred_kmeans = model.predict(vector_space)

    acc_score = 0
    data_y = [None]*number_samples
    for idx, single_training_data_y in enumerate(training_data_y):
        data_y[idx] = np.argmax(single_training_data_y)
        if np.argmax(single_training_data_y) == pred_kmeans[idx]:
            acc_score += 1
    acc_score /= number_samples
    print(pred_kmeans)
    print(acc_score)

    set_id = model.predict(vector_space[sample_id].reshape(1, -1))

    print('when the model saw')
    plt.imshow(training_data[sample_id].reshape((28, 28)), cmap='gray', interpolation='none')
    plt.show()
    print('it was similar to explanation sample set no '+str(set_id))
    count = 0
    for idx, entry in enumerate(pred_kmeans):
        if entry == set_id:
            plt.imshow(training_data[idx].reshape((28, 28)), cmap='gray', interpolation='none')
            plt.show()
            count += 1
            if count >= number_explantation_sets:
                break
    return 'done'


####not used

def make_explantation_sets(features, model, number_additional_layers,number_explantation_sets):
    '''
    not used!
    calculates the norm between the activations of the sample with every training sample and splits them in number_explantation_sets
    '''

    distance_array = np.empty([np.shape(features)[0], 2])
    layer_outputs = get_all_layer_outputs(model, features)

    for idx, _ in enumerate(features):
        sample_distance = 0
        for i in range(number_additional_layers):
            sample_distance += LA.norm(layer_outputs[i][idx:idx + 1])
        distance_array[idx, 0] = idx
        distance_array[idx, 1] = sample_distance
    distance_array = distance_array[np.argsort(distance_array[:, 1])]
    explantation_sets = np.array_split(distance_array, number_explantation_sets)
    return explantation_sets


def generate_explanation_from_sets(training_data, trained_model, number_explantation_sets, sample):
    '''
    not used!
    generates explanations from explantation_sets
    '''
    explantation_sets = make_explantation_sets(training_data, trained_model, number_explantation_sets)
    score = make_explantation_sets(sample, trained_model, 1)[0][0][1]

    explantation_set_id = -1
    for idx, explantation_set in enumerate(explantation_sets):
        if score >= explantation_set[0, 1] and score <= explantation_set[-1, 1]:
            explantation_set_id = idx
    if explantation_set_id == -1:
        return "Can not explain decision with training data"
    else:
        mask = list(map(int, explantation_sets[explantation_set_id][:, 0]))
        explantation_samples = training_data[mask]
        return_message = "When the model 'saw' \n" \
                         + str(sample[0]) \
                         + "\n it was 'thinking' the same as when it 'saw' \n" \
                         + str(explantation_samples)
        return return_message

