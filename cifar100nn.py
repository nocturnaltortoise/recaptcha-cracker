import pickle
import numpy as np
import keras
from keras.utils import np_utils
from keras.datasets import cifar100
import os.path

# all this dataset loading is all cool and stuff, but it's flat, and reshaping is irritating,
# plus keras has a properly shaped version

# def load_datasets(files):
#     datasets = {}
#     for file in files:
#         with open(file, 'rb') as f:
#             dict = pickle.load(f, encoding='bytes')
#             if 'meta' not in datasets:
#                 datasets['meta'] = dict
#             elif 'train' not in datasets:
#                 datasets['train'] = dict
#             else:
#                 datasets['test'] = dict
#             # filenames should be passed as a list meta, train, test
#     return datasets

keras.backend.set_image_dim_ordering('th')
seed = 7
np.random.seed(seed)

(train_data, train_labels), (test_data, test_labels) = cifar100.load_data(label_mode='fine')

normalised_train_data = train_data.astype('float32') / 255.0
normalised_test_data = test_data.astype('float32') / 255.0

one_hot_train_labels = np_utils.to_categorical(train_labels)
one_hot_test_labels = np_utils.to_categorical(test_labels)

num_classes = one_hot_test_labels.shape[1]


def compile_network(model):
    epochs = 25
    learning_rate = 0.01
    decay = learning_rate / epochs
    sgd = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())
    return model


def train_network(model, num_epochs):
    model = compile_network(model)
    model.fit(normalised_train_data, one_hot_train_labels, validation_data=(normalised_test_data, one_hot_test_labels), nb_epoch=num_epochs, batch_size=32)

    model.save_weights('initial-conv-net-weights.h5')
    return model


def evaluate_network(model):
    scores = model.evaluate(normalised_test_data, one_hot_test_labels, verbose=0)
    print(scores)


if os.path.exists('initial-conv-net.yaml'):
    with open('initial-conv-net.yaml', 'r') as f:
        model_yaml = f.read()
        loaded_model = keras.models.model_from_yaml(model_yaml)
        if os.path.exists('initial-conv-net-weights.h5'):
            loaded_model.load_weights('initial-conv-net-weights.h5')
            compile_network(loaded_model)
            evaluate_network(loaded_model)
        else:
            epochs = 25
            trained_model = train_network(loaded_model, epochs)
            evaluate_network(trained_model)
else:
    model = keras.models.Sequential()
    model.add(keras.layers.Convolution2D(32, 3, 3, input_shape=(3, 32, 32), border_mode='same', activation='relu', W_constraint=keras.constraints.maxnorm(3)))
    # layer for filtering 2d images
    # input shape - 32x32 images, each with three channels (i.e. RGB)
    model.add(keras.layers.Dropout(0.2))  # 20% dropout
    model.add(keras.layers.Convolution2D(32, 3, 3, border_mode='same', activation='relu', W_constraint=keras.constraints.maxnorm(3)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # think this is some kind of feature reduction layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu', W_constraint=keras.constraints.maxnorm(3))) # fully connected layer
    model.add(keras.layers.Dropout(0.5)) # 50% dropout
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    # softmax layer as output so we can get probability distribution of classes

    model_yaml = model.to_yaml()

    with open('initial-conv-net.yaml', 'w') as f:
        f.write(model_yaml)

    epochs = 25
    trained_model = train_network(model, epochs)
    evaluate_network(trained_model)

