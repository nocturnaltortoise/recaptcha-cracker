import pickle
import numpy as np
import keras
from keras.utils import np_utils
import os.path
import skimage.io
import glob
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from PIL import Image
import random
import math


def resize_images(paths):
    print("resizing images")
    for path in paths:
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            filename, ext = os.path.splitext(path)
            if not os.path.exists(filename + "_110x110" + ext) or os.path.getsize(filename + "_110x110" + ext) == 0:
                # if the file hasn't been resized or the resized version is corrupt (i.e. zero size)
                if "_110x110" not in filename:
                    # print("resizing: {0}".format(filename + ext))
                    try:
                        image = Image.open(path)
                        image = image.resize((110, 110))
                        image.save(filename + "_110x110" + ext)
                    except OSError:
                        print("OSError caused by file at {0}".format(path))
                        continue  # if OSError: cannot identify image file occurs despite above checks, skip the image


def colour_images(paths):
    print("colouring images")
    for path in paths:
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            filename, ext = os.path.splitext(path)
            if "_110x110" in filename:
                # print("opening: {0}".format(filename + ext))
                try:
                    image = Image.open(path)
                    if image.mode != "RGB":
                        print("image not RGB, colouring")
                        image = image.convert("RGB")
                        image.save(filename + ext)
                except OSError:
                    print("OSError caused by file at {0}".format(path))
                    continue  # if OSError: cannot identify image file occurs despite above checks, skip the image


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
    images = skimage.io.imread_collection('*_110x110.jpg')
    image_array = skimage.io.concatenate_images(images)
    # image_array = np.transpose(image_array, (0, 3, 1, 2)) # reorder to fit training data

    if model:
        top_n = 5
        images_predictions = model.predict_proba(image_array)
        all_predictions = [[i for i, pred in sorted([(i, pred) for i, pred in enumerate(image_predictions) if pred > 0], key=lambda x:x[1], reverse=True)[:top_n]] for image_predictions in images_predictions]
        return all_predictions


# def train_network(model, num_chunks):
#     print("Training network.")
#     model = compile_network(model, num_epochs)
#     #  model.fit_generator(normalise_and_convert_to_one_hot_train(num_chunks), samples_per_epoch=72000, nb_epoch=num_epochs, validation_data=normalise_and_convert_to_one_hot_validation(num_chunks), nb_val_samples=2000)
#     train, train_labels = next(normalise_and_convert_to_one_hot_train(num_chunks))
#     validation, validation_labels = next(normalise_and_convert_to_one_hot_validation(num_chunks))
#     model.fit(train, train_labels, batch_size=32, nb_epoch=1, validation_data=(validation, validation_labels))
#     model.save_weights('model-6-conv-net-weights.h5')


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
    learning_rate = 0.1
    decay = learning_rate / num_epochs
    # decay = 0.5
    sgd = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, decay=1e-6, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])
    # print(model.summary())
    return model


def read_labels(path):
    print("reading labels in {0}".format(path))
    labels = []
    filenames = []
    with open(path, 'r') as f:
        for line in f:
            filename, label = line.split(" ")
            labels.append(int(label.rstrip('\r\n')))
            filenames.append(filename)

    return filenames, labels


def convert_to_one_hot(labels):
    print("converting to one hot")
    return np_utils.to_categorical(labels)


def normalise(image_data):
    # image_data = image_data.astype('float') - np.mean(image_data, axis=0)  # subtract mean
    # image_data = image_data / np.std(image_data, axis=0)  # divide by standard deviation
    return image_data.astype('float') / 255.0


def process_filepaths(paths, root_path, remove_leading_slash=True):
    new_paths = []
    if remove_leading_slash:
        for path in paths:
            full_path = os.path.join(root_path, path[1:])
            if os.path.isfile(full_path) and os.path.getsize(full_path) > 0:
                new_paths.append(full_path)
        # filenames for training images have a leading slash, which causes problems on windows with os.path.join
    else:
        for path in paths:
            full_path = os.path.join(root_path, path)
            if os.path.isfile(full_path) and os.path.getsize(full_path) > 0:
                new_paths.append(full_path)

    return new_paths


def change_filepaths_after_resize(paths):
    resize_paths = []
    for path in paths:
        if "_110x110" not in path:
            name, ext = os.path.splitext(path)
            path = name + "_110x110" + ext
        resize_paths.append(path)

    return resize_paths


def create_network():
    num_classes = train_labels.shape[1]
    img_input = (110, 110, 3)
    model = keras.models.Sequential()
    # Block 1
    model.add(keras.layers.Convolution2D(32, 3, 3, input_shape=img_input, activation='relu', border_mode='same', name='block1_conv1'))
    model.add(keras.layers.Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='block1_conv2'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # # Block 2
    # model.add(keras.layers.Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1'))
    # model.add(keras.layers.Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2'))
    # model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    #
    # # Block 3
    # model.add(keras.layers.Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1'))
    # model.add(keras.layers.Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2'))
    # model.add(keras.layers.Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3'))
    # model.add(keras.layers.Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv4'))
    # model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    #
    # # Block 4
    # model.add(keras.layers.Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1'))
    # model.add(keras.layers.Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2'))
    # model.add(keras.layers.Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3'))
    # model.add(keras.layers.Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv4'))
    # model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
    #
    # # Block 5
    # model.add(keras.layers.Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1'))
    # model.add(keras.layers.Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2'))
    # model.add(keras.layers.Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3'))
    # model.add(keras.layers.Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv4'))
    # model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    # Classification block
    model.add(keras.layers.Flatten(name='flatten'))
    model.add(keras.layers.Dense(512, activation='relu', name='fc1'))
    # model.add(keras.layers.Dense(4096, activation='relu', name='fc2'))
    model.add(keras.layers.Dense(num_classes, activation='softmax', name='predictions'))

    # model_yaml = model.to_yaml()

    # with open('model-6-conv-net.yaml', 'w') as f:
    #     f.write(model_yaml)

    return model


def next_train_batch():
    chunk_size = 23
    # for i in range(num_train_chunks):
    i = 0
    while True:
        print("loading train chunk {0}".format(i / chunk_size))
        chunk_filepaths = process_filepaths(train_files[i:i + chunk_size], 'E:/datasets/data_256/')
        resize_images(chunk_filepaths)
        chunk_filepaths = change_filepaths_after_resize(chunk_filepaths)
        colour_images(chunk_filepaths)
        chunk_images = normalise(skimage.io.imread_collection(chunk_filepaths).concatenate())
        chunk_labels = train_labels[i:i + chunk_size]
        yield chunk_images, chunk_labels
        i += chunk_size
        if i + chunk_size > train_size:
            i = 0


def next_validation_batch():
    chunk_size = 20
    # for i in range(num_val_chunks):
    i = 0
    while True:
        print("loading validation chunk {0}".format(i))
        chunk_filepaths = process_filepaths(validation_files[i:i + chunk_size], 'E:/datasets/val_256/', remove_leading_slash=False)
        resize_images(chunk_filepaths)
        chunk_filepaths = change_filepaths_after_resize(chunk_filepaths)
        colour_images(chunk_filepaths)
        chunk_images = normalise(skimage.io.imread_collection(chunk_filepaths).concatenate())
        chunk_labels = validation_labels[i:i + chunk_size]
        yield chunk_images, chunk_labels
        i += chunk_size
        if i + chunk_size > validation_size:
            i = 0


num_epochs = 46

# seed = 123124
# np.random.seed(seed)

train_files, train_labels = read_labels('../../datasets/places365_train_standard.txt')
train_files, test_files, train_labels, test_labels = train_test_split(train_files, train_labels, test_size=0.2, random_state=2134712)
train_labels = convert_to_one_hot(train_labels)

# # files_and_labels = list(zip(train_files, train_labels))
# # random.shuffle(files_and_labels)
# # train_files[:], train_labels[:] = zip(*files_and_labels)

validation_files, validation_labels = read_labels('../../datasets/places365_val.txt')
validation_labels = convert_to_one_hot(validation_labels)
# # validation_labels = np.array(validation_labels)
# # test_files, test_labels = read_labels('../../datasets/places365_test.txt')
# # test_labels = convert_to_one_hot(test_labels)
#
#
train_size = len(train_files)
print(train_size)
# num_train_chunks = int(train_size / 23) * num_epochs # reasonable size that divides train set evenly
validation_size = len(validation_files)
# num_val_chunks = validation_size
#
# # train_chunk_size = int(train_size / num_train_chunks)
# # validation_chunk_size = int(validation_size / num_val_chunks)
# # print(train_size, validation_size)
#
model = create_network()
model = compile_network(model, num_epochs)
#
tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
checkpointer = keras.callbacks.ModelCheckpoint(filepath="generator-model-conv-net-weights.h5", verbose=1, save_best_only=True)
model.fit_generator(next_train_batch(), samples_per_epoch=int(train_size / (num_epochs / 2)), nb_epoch=num_epochs, validation_data=next_validation_batch(), nb_val_samples=int(validation_size / (num_epochs / 2)), callbacks=[checkpointer, tensorboard])
#

# model.save_weights('generator-model-conv-net-weights.h5')
# for i in range(num_epochs):
#     for j in range(num_chunks):
#         chunk_images, chunk_labels = next(next_batch("train", train_files, train_labels,
#                                                      validation_files, validation_labels,
#                                                      j, train_chunk_size))
#
#         train_chunk_images, test_chunk_images, train_chunk_labels, test_chunk_labels = train_test_split(chunk_images, chunk_labels, test_size=0.2, random_state=123412)
#         print(train_chunk_images.shape, train_chunk_labels.shape)
#         # model.train_on_batch(train_chunk_images, train_chunk_labels)
#         val_chunk_images, val_chunk_labels = next(next_batch("validation", train_files, train_labels,
#                                                          validation_files, validation_labels,
#                                                          j, validation_chunk_size))
#         print(val_chunk_images.shape, val_chunk_labels.shape)
#
#         model.fit(train_chunk_images, train_chunk_labels, batch_size=32, nb_epoch=1, validation_data=(val_chunk_images, val_chunk_labels))

#     if os.path.exists('model-6-conv-net.yaml'):
#         print("Loading model from file.")
#         with open('model-6-conv-net.yaml', 'r') as f:
#             model_yaml = f.read()
#             loaded_model = keras.models.model_from_yaml(model_yaml)
#             # if os.path.exists('model-6-conv-net-weights.h5'):
#             print("Loading weights from file.")
#             loaded_model.load_weights('model-6-conv-net-weights.h5')
#             train_network(loaded_model, num_chunks)
#             # else:
#             #     print("Model exists but weights do not, training network.")
#             #     train_network(loaded_model, normalised_train_data, one_hot_train_labels)
#     else:
#         # test, test_labels = next(normalise_and_convert_to_one_hot_train(num_chunks))
#
#         model_yaml = model.to_yaml()
#
#         with open('model-6-conv-net.yaml', 'w') as f:
#             f.write(model_yaml)
#
#         train_network(model, num_chunks)






