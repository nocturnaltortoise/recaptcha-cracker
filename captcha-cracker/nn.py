import keras
import skimage.io
from sklearn.model_selection import train_test_split
from preprocessors import ImagePreprocessor, FilepathPreprocessor, LabelProcessor
import numpy as np
import os.path
from config import config
import keras.applications.xception
import keras.applications.vgg19


class NeuralNetwork:

    def __init__(self, learning_rate=0.01, decay_rate=1e-6, weights_file=None, continue_training=False, start_epoch=None):
        if not weights_file or continue_training:
            self.num_epochs = 10
            self.train_files, self.train_labels = LabelProcessor.read_labels([config['labels_path']])
            self.train_files, self.validation_files, self.train_labels, self.validation_labels = train_test_split(self.train_files,
                                                                                                      self.train_labels,
                                                                                                      test_size=0.1,
                                                                                                      random_state=2134)
            self.train_labels = LabelProcessor.convert_to_one_hot(self.train_labels)
            self.validation_labels = LabelProcessor.convert_to_one_hot(self.validation_labels)
            self.train_size = len(self.train_files)
            self.validation_size = len(self.validation_files)

        if weights_file is not None and continue_training:
            self.model = self.xception(include_top=True)
            if os.path.exists(weights_file):
                self.model.load_weights(weights_file)
                self.compile_network(learning_rate, decay_rate)
                self.train_network(start_epoch)
        elif weights_file is not None:
            self.model = self.xception(include_top=True)
            if os.path.exists(weights_file):
                self.model.load_weights(weights_file)
                self.compile_network(learning_rate, decay_rate)
        else:
            self.model = self.xception(include_top=True)
            self.compile_network(learning_rate, decay_rate)
            self.train_network(learning_rate, decay_rate)


    def predict_image_classes(self, checkboxes):
        image_paths = [checkbox.image_path for checkbox in checkboxes]
        image_paths = FilepathPreprocessor.change_filepaths_after_resize(image_paths)
        images = skimage.io.imread_collection(image_paths)
        image_array = skimage.io.concatenate_images(images)
        image_array = ImagePreprocessor.normalise(image_array)

        probability_threshold = 0.01
        images_predictions = self.model.predict(image_array)
        all_predictions = []
        for image_predictions in images_predictions:
            individual_predictions = []
            for i, probability in enumerate(image_predictions):
                if probability > probability_threshold:
                    individual_predictions.append((i, probability))

            all_predictions.append(individual_predictions)

        labels = []
        for individual_predictions in all_predictions:
            sorted_predictions = sorted(individual_predictions, key=lambda x:x[1], reverse=True)
            image_labels = []
            for (class_label, probability) in sorted_predictions:
                image_labels.append(class_label)
            labels.append(image_labels)

        return labels

    def compile_network(self, learning_rate, decay_rate):
        learning_rate = learning_rate
        decay = decay_rate
        # adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        sgd = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, decay=decay, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd,
                      metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])

    def xception(self, include_top=True):
        # num_classes = self.train_labels.shape[1]
        num_classes = config['num_classes']
        size = config['image_size_tuple']
        width = size[0]
        height = size[1]
        img_input = keras.layers.Input(shape=(size[0], size[1], 3))

        # if include_top:
        #     model = keras.applications.vgg19.VGG19(
        #         include_top=True, 
        #         weights=None, 
        #         input_tensor=img_input,
        #         classes=num_classes)
        # else:
        #     model = keras.applications.vgg19.VGG19(
        #         include_top=False, 
        #         weights=None, 
        #         input_tensor=img_input,
        #         pooling='avg')
        if include_top:
            model = keras.applications.xception.Xception(
                include_top=True, 
                weights=None, 
                input_tensor=img_input,
                classes=num_classes)
        else:
            model = keras.applications.xception.Xception(
                include_top=False, 
                weights=None, 
                input_tensor=img_input,
                pooling='avg')

        return model

    def next_train_batch(self, chunk_size):
        i = 0
        while True:
            print("loading train chunk {0}".format(i / chunk_size))
            chunk_filepaths = FilepathPreprocessor.process_filepaths(self.train_files[i:i + chunk_size],
                                                                     [config['dataset_path']])
            ImagePreprocessor.resize_images(chunk_filepaths)
            chunk_filepaths = FilepathPreprocessor.change_filepaths_after_resize(chunk_filepaths)
            ImagePreprocessor.colour_images(chunk_filepaths)
            chunk_images = ImagePreprocessor.normalise(skimage.io.imread_collection(chunk_filepaths).concatenate())
            chunk_labels = self.train_labels[i:i + chunk_size]
            yield chunk_images, chunk_labels
            i += chunk_size
            if i + chunk_size > self.train_size:
                i = 0

    def next_validation_batch(self, chunk_size):
        i = 0
        while True:
            print("loading validation chunk {0}".format(i / chunk_size))
            chunk_filepaths = FilepathPreprocessor.process_filepaths(self.validation_files[i:i + chunk_size],
                                                                     [config['dataset_path']])
            ImagePreprocessor.resize_images(chunk_filepaths)
            chunk_filepaths = FilepathPreprocessor.change_filepaths_after_resize(chunk_filepaths)
            ImagePreprocessor.colour_images(chunk_filepaths)
            chunk_images = ImagePreprocessor.normalise(skimage.io.imread_collection(chunk_filepaths).concatenate())
            chunk_labels = self.validation_labels[i:i + chunk_size]
            yield chunk_images, chunk_labels
            i += chunk_size
            if i + chunk_size > self.validation_size:
                i = 0

    def train_network(self, learning_rate, decay_rate, start_epoch=None):
        # weights_path = "weights/{0}-rerun-test.h5".format(learning_rate)
        # log_path = "logs/{0}-rerun-test".format(learning_rate)
        tensorboard = keras.callbacks.TensorBoard(log_dir=config['log_path'],
                                                  histogram_freq=0,
                                                  write_graph=True,
                                                  write_images=False)
        checkpointer = keras.callbacks.ModelCheckpoint(filepath=config['weights_path'],
                                                       verbose=1,
                                                       save_best_only=True)
        self.model.fit_generator(self.next_train_batch(chunk_size=16),
                                 samples_per_epoch=int(self.train_size/5),
                                 nb_epoch=self.num_epochs,
                                 validation_data=self.next_validation_batch(chunk_size=32),
                                 nb_val_samples=int(self.validation_size/5),
                                 callbacks=[checkpointer, tensorboard])

# learning_rate = 0.001
# decay_rate = 0.001
# neural_net = NeuralNetwork(learning_rate, decay_rate)