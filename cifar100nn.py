import pickle
import numpy as np
import keras
from keras.utils import np_utils
import os.path
import skimage.io
import glob
from PIL import Image
import pickle


# def load_labels():
#     with open('datasets/cifar-100-python/meta', 'rb') as f:
#         label_names = pickle.load(f)['fine_label_names']
#         label_names.append('street numbers')
#         return label_names
#
#
def convert_labels_to_label_names(labels):
    print(labels)
    selected_images_names = [folder for folder in os.listdir('datasets/selected_images/')]
    print(selected_images_names)
    other_names = ['street numbers', 'street signs']
    names = selected_images_names + other_names
    print(names)
    label_names = [names[label] for label in labels]
    print(labels, label_names)
    return label_names


def load_npy_data():
    print("Loading train dataset.")
    train = np.load('datasets/train.npy')
    print("Loading test dataset.")
    test = np.load('datasets/test.npy')
    print("Loading train labels.")
    train_labels = np.load('datasets/train_labels.npy')
    print("Loading test labels.")
    test_labels = np.load('datasets/test_labels.npy')
    print("Loading validation dataset")
    validation = np.load('datasets/validation.npy')
    print("Loading validation labels.")
    validation_labels = np.load('datasets/validation_labels.npy')

    return (train, train_labels), (test, test_labels), (validation, validation_labels)


def load_model():
    if os.path.exists('model-3-conv-net.yaml'):
        print("Loading model from file.")
        with open('model-3-conv-net.yaml', 'r') as f:
            model_yaml = f.read()
            loaded_model = keras.models.model_from_yaml(model_yaml)
            if os.path.exists('model-3-conv-net-weights.h5'):
                print("Loading weights from file.")
                loaded_model.load_weights('model-3-conv-net-weights.h5')
                epochs = 25
                model = compile_network(loaded_model, epochs)
    return model


def predict_image_classes():
    model = load_model()
    images = skimage.io.imread_collection('*_32x32.jpg')
    image_array = skimage.io.concatenate_images(images)
    # image_array = np.transpose(image_array, (0, 3, 1, 2)) # reorder to fit training data

    if model:
        predictions = model.predict_classes(image_array)
        return predictions


def compile_network(model, num_epochs):
    print("Compiling network.")
    learning_rate = 0.01
    decay = learning_rate / num_epochs
    sgd = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # print(model.summary())
    return model

#
# keras.backend.set_image_dim_ordering('th')
seed = 7
np.random.seed(seed)

(train_data, train_labels), (test_data, test_labels), (validation_data, validation_labels) = load_npy_data()
# # (cifar_train_data, cifar_train_labels), (cifar_test_data, cifar_test_labels) = cifar100.load_data(label_mode='fine')
# # print(cifar_train_data.shape, train_data.shape)
#
print("normalising train data")
normalised_train_data = train_data.astype('float32') / 255.0
print("normalising test data")
normalised_test_data = test_data.astype('float32') / 255.0
print("normalising validation data")
normalised_validation_data = validation_data.astype('float32') / 255.0

print("converting train labels to one hot")
one_hot_train_labels = np_utils.to_categorical(train_labels)
print(len(np.unique(train_labels)))
print("converting test labels to one hot")
one_hot_test_labels = np_utils.to_categorical(test_labels)
print(len(np.unique(test_labels)))
print("converting validation labels to one hot")
one_hot_validation_labels = np_utils.to_categorical(validation_labels)


def train_network(model, num_epochs):
    print("Training network.")
    model = compile_network(model, num_epochs)
    model.fit(normalised_train_data, one_hot_train_labels, validation_data=(normalised_validation_data, one_hot_validation_labels),
              nb_epoch=num_epochs, batch_size=32)

    model.save_weights('model-3-conv-net-weights.h5')


def evaluate_network(model):
    print("Evaluating network.")
    scores = model.evaluate(normalised_test_data, one_hot_test_labels, verbose=0)
    print(scores)


# if os.path.exists('model-3-conv-net.yaml'):
#     print("Loading model from file.")
#     with open('model-3-conv-net.yaml', 'r') as f:
#         model_yaml = f.read()
#         loaded_model = keras.models.model_from_yaml(model_yaml)
#         if os.path.exists('model-3-conv-net-weights.h5'):
#             print("Loading weights from file.")
#             loaded_model.load_weights('model-3-conv-net-weights.h5')
#             epochs = 25
#             compile_network(loaded_model, epochs)
#             evaluate_network(loaded_model)
#         else:
#             print("Model exists but weights do not, training network.")
#             epochs = 25
#             train_network(loaded_model, epochs)
#             evaluate_network(loaded_model)
# else:
#     num_classes = one_hot_test_labels.shape[1]
#     # the 2 is for the svhn and gtrsb datasets, both have a single label
#     model = keras.models.Sequential()
#     model.add(keras.layers.Convolution2D(32, 32, 3, input_shape=(32, 32, 3), border_mode='same', activation='relu',
#                                          W_constraint=keras.constraints.maxnorm(3)))
#     model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(keras.layers.Convolution2D(32, 32, 3, border_mode='same', activation='relu',
#                                          W_constraint=keras.constraints.maxnorm(3)))
#     model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(keras.layers.Flatten())
#     model.add(keras.layers.Dense(512, activation='relu', W_constraint=keras.constraints.maxnorm(3)))
#     model.add(keras.layers.Dropout(0.5))  # 50% dropout
#     model.add(keras.layers.Dense(num_classes, activation='softmax'))
#     # softmax layer as output so we can get probability distribution of classes
#
#     model_yaml = model.to_yaml()
#
#     with open('model-3-conv-net.yaml', 'w') as f:
#         f.write(model_yaml)
#
#     epochs = 25
#     train_network(model, epochs)
#     evaluate_network(model)
