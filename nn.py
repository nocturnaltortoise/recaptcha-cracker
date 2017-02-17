import pickle
import numpy as np
import keras
from keras.utils import np_utils
import os.path
import skimage.io
import glob
from PIL import Image
import pickle
import matplotlib.pyplot as plt


def convert_labels_to_label_names(labels):
    # print(labels)
    # selected_images_names = [folder for folder in os.listdir('datasets/selected_images/')]
    # print(selected_images_names)
    # other_names = ['street numbers', 'street signs']
    chosen_label_names = []
    for image_labels in labels:
        names = np.load('names.npy')

        label_names = [names[label] for label in image_labels]

        label_categories = {
            'house': 'a house',
            'beach_house': 'a house',
            'boathouse': 'a house',
            'fastfood_restaurant': 'store front',
            'gas_station': 'store front',
            'general_store': 'store front',
            'oast_house': 'a house',
            'shopfront': 'store front',
            'storefront': 'store front',
            'mansion': 'a house'
        }

        for i, label_name in enumerate(label_names):
            if label_name in label_categories:
                label_names[i] = label_categories[label_name]
            else:
                label_names[i] = label_names[i].replace("_"," ")

        chosen_label_names.append(label_names)

    return chosen_label_names



def load_model():
    if os.path.exists('model-6-conv-net.yaml'):
        print("Loading model from file.")
        with open('model-6-conv-net.yaml', 'r') as f:
            model_yaml = f.read()
            loaded_model = keras.models.model_from_yaml(model_yaml)
            if os.path.exists('model-6-conv-net-weights.h5'):
                print("Loading weights from file.")
                loaded_model.load_weights('model-6-conv-net-weights.h5')
                epochs = 10
                model = compile_network(loaded_model, epochs)
    return model


def predict_image_classes():
    model = load_model()
    images = skimage.io.imread_collection('*_32x32.jpg')
    image_array = skimage.io.concatenate_images(images)
    # image_array = np.transpose(image_array, (0, 3, 1, 2)) # reorder to fit training data

    if model:
        top_n = 5
        images_predictions = model.predict_proba(image_array)
        all_predictions = [[i for i, pred in sorted([(i, pred) for i, pred in enumerate(image_predictions) if pred > 0], key=lambda x:x[1], reverse=True)[:top_n]] for image_predictions in images_predictions]
        return all_predictions


def train_network(model, num_chunks):
    print("Training network.")
    model = compile_network(model, num_epochs)
    #  model.fit_generator(normalise_and_convert_to_one_hot_train(num_chunks), samples_per_epoch=72000, nb_epoch=num_epochs, validation_data=normalise_and_convert_to_one_hot_validation(num_chunks), nb_val_samples=2000)
    train, train_labels = next(normalise_and_convert_to_one_hot_train(num_chunks))
    validation, validation_labels = next(normalise_and_convert_to_one_hot_validation(num_chunks))
    model.fit(train, train_labels, batch_size=32, nb_epoch=1, validation_data=(validation, validation_labels))
    model.save_weights('model-6-conv-net-weights.h5')


# def evaluate_network(model):
#     print("Evaluating network.")
#     scores = model.evaluate(normalised_test_data, one_hot_test_labels, verbose=0)
#     print(scores)


def accuracy_graph(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def loss_graph(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def accuracy_loss_graph(history):
    plt.plot(history.history['loss'], history.history['acc'])
    plt.plot(history.history['val_loss'], history.history['val_acc'])
    plt.title('Accuracy against Loss')
    plt.ylabel('Accuracy')
    plt.xlabel('Loss')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def compile_network(model, num_epochs):
    print("Compiling network.")
    learning_rate = 0.01
    decay = learning_rate / num_epochs
    sgd = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # print(model.summary())
    return model

num_chunks = 137  # can we calculate this somehow?
num_epochs = 25
# for i in range(num_epochs):
    # for
seed = 7
np.random.seed(seed)


def find_image_data(train_path, validation_path, test_path):
    train_filepaths = [path for path in glob.glob(train_path) if os.path.isfile(path)]
    test_filepaths = [path for path in glob.glob(test_path) if os.path.isfile(path)]
    validation_filepaths = [path for path in glob.glob(validation_path) if os.path.isfile(path)]

    return train_filepaths, validation_filepaths, test_filepaths


def next_batch(type, train_filepaths, validation_filepaths, test_filepaths):
    # select a chunk of filepaths, load those images, and their labels, yield the chunk
    if type == "train":
        pass
    elif type == "validation":
        pass
    else:
        pass


def normalise_and_convert_to_one_hot_train(num_chunks):
    for (train_data, train_labels), (test_data, test_labels), (validation_data, validation_labels) in load_npy_data(num_chunks):
        print("normalising train data")
        normalised_train_data = train_data.astype('float32') / 255.0

        print("converting train labels to one hot")
        one_hot_train_labels = np_utils.to_categorical(train_labels)
        print(len(np.unique(train_labels)))
        yield (normalised_train_data, one_hot_train_labels)


def normalise_and_convert_to_one_hot_test(num_chunks):
    for (train_data, train_labels), (test_data, test_labels), (validation_data, validation_labels) in load_npy_data(num_chunks):
        print("normalising test data")
        normalised_test_data = test_data.astype('float32') / 255.0

        print("converting test labels to one hot")
        one_hot_test_labels = np_utils.to_categorical(test_labels)
        print(len(np.unique(test_labels)))
        print(one_hot_test_labels.shape)
        yield (normalised_test_data, one_hot_test_labels)


def normalise_and_convert_to_one_hot_validation(num_chunks):
    for (train_data, train_labels), (test_data, test_labels), (validation_data, validation_labels) in load_npy_data(num_chunks):
        print("normalising validation data")
        normalised_validation_data = validation_data.astype('float32') / 255.0

        print("converting validation labels to one hot")
        one_hot_validation_labels = np_utils.to_categorical(validation_labels)
        yield (normalised_validation_data, one_hot_validation_labels)


for i in range(num_epochs):
    if os.path.exists('model-6-conv-net.yaml'):
        print("Loading model from file.")
        with open('model-6-conv-net.yaml', 'r') as f:
            model_yaml = f.read()
            loaded_model = keras.models.model_from_yaml(model_yaml)
            # if os.path.exists('model-6-conv-net-weights.h5'):
            print("Loading weights from file.")
            loaded_model.load_weights('model-6-conv-net-weights.h5')
            train_network(loaded_model, num_chunks)
            # else:
            #     print("Model exists but weights do not, training network.")
            #     train_network(loaded_model, normalised_train_data, one_hot_train_labels)
    else:
        # test, test_labels = next(normalise_and_convert_to_one_hot_train(num_chunks))
        num_classes = 365


        model = keras.models.Sequential()
        model.add(keras.layers.Convolution2D(20, 5, 5,
                                            input_shape=(110, 110, 3),
                                            border_mode='same',
                                            activation='relu',
                                            W_constraint=keras.constraints.maxnorm(3)))
        model.add(keras.layers.Convolution2D(20, 3, 3,
                                            border_mode='same',
                                            activation='relu',
                                            W_constraint=keras.constraints.maxnorm(3)))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Convolution2D(20, 3, 3,
                                            border_mode='same',
                                            activation='relu',
                                            W_constraint=keras.constraints.maxnorm(3)))
        model.add(keras.layers.Convolution2D(20, 3, 3,
                                            border_mode='same',
                                            activation='relu',
                                            W_constraint=keras.constraints.maxnorm(3)))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Convolution2D(20, 3, 3,
                                            border_mode='same',
                                            activation='relu',
                                            W_constraint=keras.constraints.maxnorm(3)))
        model.add(keras.layers.Convolution2D(20, 3, 3,
                                            border_mode='same',
                                            activation='relu',
                                            W_constraint=keras.constraints.maxnorm(3)))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Convolution2D(20, 3, 3,
                                            border_mode='same',
                                            activation='relu',
                                            W_constraint=keras.constraints.maxnorm(3)))
        model.add(keras.layers.Convolution2D(20, 3, 3,
                                            border_mode='same',
                                            activation='relu',
                                            W_constraint=keras.constraints.maxnorm(3)))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(2048, activation='relu', W_constraint=keras.constraints.maxnorm(3)))
        model.add(keras.layers.Dropout(0.5))  # 50% dropout
        model.add(keras.layers.Dense(num_classes, activation='softmax'))
        # softmax layer as output so we can get probability distribution of classes

        model_yaml = model.to_yaml()

        with open('model-6-conv-net.yaml', 'w') as f:
            f.write(model_yaml)

        train_network(model, num_chunks)






