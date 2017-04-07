import keras
import skimage.io
from sklearn.model_selection import train_test_split
from preprocessors import ImagePreprocessor, FilepathPreprocessor, LabelProcessor
import numpy as np
import os.path


class NeuralNetwork:

    def __init__(self, weights_file=None, continue_training=False, start_epoch=None):
        self.num_epochs = 100
        self.train_files, self.train_labels = LabelProcessor.read_labels(['../../datasets/captcha-dataset-labels.txt'])
        self.train_files, self.test_files, self.train_labels, self.test_labels = train_test_split(self.train_files,
                                                                                                  self.train_labels,
                                                                                                  test_size=0.1,
                                                                                                  random_state=2134)
        self.train_files, self.validation_files, self.train_labels, self.validation_labels = train_test_split(self.train_files,
                                                                                                              self.train_labels,
                                                                                                              test_size=0.2,
                                                                                                              random_state=124)
        print("unique train labels: ", len(np.unique(self.train_labels)))
        self.train_labels = LabelProcessor.convert_to_one_hot(self.train_labels)
        self.validation_labels = LabelProcessor.convert_to_one_hot(self.validation_labels)
        self.train_size = len(self.train_files)
        print(self.train_size)
        self.validation_size = len(self.validation_files)
        print(self.validation_size)

        if weights_file is not None and continue_training:
            self.model = self.xception(include_top=True)
            if os.path.exists(weights_file):
                self.model.load_weights(weights_file)
                self.compile_network()
                self.train_network(start_epoch)
        elif weights_file is not None:
            self.model = self.xception(include_top=True)
            if os.path.exists(weights_file):
                self.model.load_weights(weights_file)
                self.compile_network()
        else:
            self.model = self.xception(include_top=True)
            self.compile_network()
            self.train_network()


    def predict_image_classes(self):
        images = skimage.io.imread_collection('*_110x110.jpg')
        image_array = skimage.io.concatenate_images(images)
        image_array = ImagePreprocessor.normalise(image_array)

        top_n = 5
        images_predictions = self.model.predict(image_array)
        all_predictions = []
        for image_predictions in images_predictions:
            individual_predictions = []
            for i, probability in enumerate(image_predictions):
                if probability > 0.1:
                    individual_predictions.append((i,probability))

            all_predictions.append(individual_predictions)

        all_predictions = [[class_label for (class_label, probability) in sorted(individual_predictions, key=lambda x:x[1], reverse=True)] for individual_predictions in all_predictions]

        # print(all_predictions)
        return all_predictions

    def compile_network(self):
        print("Compiling network.")
        learning_rate = 0.01
        decay = 1e-6
        sgd = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, decay=decay, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd,
                      metrics=['categorical_accuracy'])

    def xception(self, include_top=True):
        if os.path.exists('xception-less-data-model.yaml'):
            with open('xception-less-data-model.yaml', 'r') as f:
                loaded_model_yaml = f.read()
                return keras.models.model_from_yaml(loaded_model_yaml)
        else:
            num_classes = self.train_labels.shape[1]
            img_input = keras.layers.Input(shape=(93, 93, 3))

            x = keras.layers.Conv2D(32, 3, 3, subsample=(2, 2), bias=False, name='block1_conv1')(img_input)
            x = keras.layers.BatchNormalization(name='block1_conv1_bn')(x)
            x = keras.layers.Activation('relu', name='block1_conv1_act')(x)
            x = keras.layers.Conv2D(64, 3, 3, bias=False, name='block1_conv2')(x)
            x = keras.layers.BatchNormalization(name='block1_conv2_bn')(x)
            x = keras.layers.Activation('relu', name='block1_conv2_act')(x)

            residual = keras.layers.Conv2D(128, 1, 1, subsample=(2, 2),
                                           border_mode='same', bias=False)(x)
            residual = keras.layers.BatchNormalization()(residual)

            x = keras.layers.SeparableConv2D(128, 3, 3, border_mode='same', bias=False, name='block2_sepconv1')(x)
            x = keras.layers.BatchNormalization(name='block2_sepconv1_bn')(x)
            x = keras.layers.Activation('relu', name='block2_sepconv2_act')(x)
            x = keras.layers.SeparableConv2D(128, 3, 3, border_mode='same', bias=False, name='block2_sepconv2')(x)
            x = keras.layers.BatchNormalization(name='block2_sepconv2_bn')(x)

            x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block2_pool')(x)
            x = keras.layers.merge([x, residual], mode='sum')

            residual = keras.layers.Conv2D(256, 1, 1, subsample=(2, 2),
                                           border_mode='same', bias=False)(x)
            residual = keras.layers.BatchNormalization()(residual)

            x = keras.layers.Activation('relu', name='block3_sepconv1_act')(x)
            x = keras.layers.SeparableConv2D(256, 3, 3, border_mode='same', bias=False, name='block3_sepconv1')(x)
            x = keras.layers.BatchNormalization(name='block3_sepconv1_bn')(x)
            x = keras.layers.Activation('relu', name='block3_sepconv2_act')(x)
            x = keras.layers.SeparableConv2D(256, 3, 3, border_mode='same', bias=False, name='block3_sepconv2')(x)
            x = keras.layers.BatchNormalization(name='block3_sepconv2_bn')(x)

            x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block3_pool')(x)
            x = keras.layers.merge([x, residual], mode='sum')

            residual = keras.layers.Conv2D(728, 1, 1, subsample=(2, 2),
                                           border_mode='same', bias=False)(x)
            residual = keras.layers.BatchNormalization()(residual)

            x = keras.layers.Activation('relu', name='block4_sepconv1_act')(x)
            x = keras.layers.SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name='block4_sepconv1')(x)
            x = keras.layers.BatchNormalization(name='block4_sepconv1_bn')(x)
            x = keras.layers.Activation('relu', name='block4_sepconv2_act')(x)
            x = keras.layers.SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name='block4_sepconv2')(x)
            x = keras.layers.BatchNormalization(name='block4_sepconv2_bn')(x)

            x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block4_pool')(x)
            x = keras.layers.merge([x, residual], mode='sum')

            for i in range(8):
                residual = x
                prefix = 'block' + str(i + 5)

                x = keras.layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
                x = keras.layers.SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name=prefix + '_sepconv1')(x)
                x = keras.layers.BatchNormalization(name=prefix + '_sepconv1_bn')(x)
                x = keras.layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
                x = keras.layers.SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name=prefix + '_sepconv2')(x)
                x = keras.layers.BatchNormalization(name=prefix + '_sepconv2_bn')(x)
                x = keras.layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
                x = keras.layers.SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name=prefix + '_sepconv3')(x)
                x = keras.layers.BatchNormalization(name=prefix + '_sepconv3_bn')(x)

                x = keras.layers.merge([x, residual], mode='sum')

            residual = keras.layers.Conv2D(1024, 1, 1, subsample=(2, 2),
                                           border_mode='same', bias=False)(x)
            residual = keras.layers.BatchNormalization()(residual)

            x = keras.layers.Activation('relu', name='block13_sepconv1_act')(x)
            x = keras.layers.SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name='block13_sepconv1')(x)
            x = keras.layers.BatchNormalization(name='block13_sepconv1_bn')(x)
            x = keras.layers.Activation('relu', name='block13_sepconv2_act')(x)
            x = keras.layers.SeparableConv2D(1024, 3, 3, border_mode='same', bias=False, name='block13_sepconv2')(x)
            x = keras.layers.BatchNormalization(name='block13_sepconv2_bn')(x)

            x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block13_pool')(x)
            x = keras.layers.merge([x, residual], mode='sum')

            x = keras.layers.SeparableConv2D(1536, 3, 3, border_mode='same', bias=False, name='block14_sepconv1')(x)
            x = keras.layers.BatchNormalization(name='block14_sepconv1_bn')(x)
            x = keras.layers.Activation('relu', name='block14_sepconv1_act')(x)

            x = keras.layers.SeparableConv2D(2048, 3, 3, border_mode='same', bias=False, name='block14_sepconv2')(x)
            x = keras.layers.BatchNormalization(name='block14_sepconv2_bn')(x)
            x = keras.layers.Activation('relu', name='block14_sepconv2_act')(x)

            if include_top:
                x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
                x = keras.layers.Dense(num_classes, activation='softmax', name='predictions')(x)

            # Create model
            model = keras.models.Model(img_input, x)

            model_yaml = model.to_yaml()
            with open('xception-less-data-model.yaml', 'w+') as f:
                f.write(model_yaml)

            return model

    def next_train_batch(self, chunk_size):
        i = 0
        while True:
            print("loading train chunk {0}".format(i / chunk_size))
            chunk_filepaths = FilepathPreprocessor.process_filepaths(self.train_files[i:i + chunk_size],
                                                                     ['E:/datasets/captcha-dataset/'])
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
                                                                     ['E:/datasets/captcha-dataset/'])
            ImagePreprocessor.resize_images(chunk_filepaths)
            chunk_filepaths = FilepathPreprocessor.change_filepaths_after_resize(chunk_filepaths)
            ImagePreprocessor.colour_images(chunk_filepaths)
            chunk_images = ImagePreprocessor.normalise(skimage.io.imread_collection(chunk_filepaths).concatenate())
            chunk_labels = self.validation_labels[i:i + chunk_size]
            yield chunk_images, chunk_labels
            i += chunk_size
            if i + chunk_size > self.validation_size:
                i = 0

    def train_network(self, start_epoch=None):
        tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/xception-less-data',
                                                  histogram_freq=0,
                                                  write_graph=True,
                                                  write_images=False)
        checkpointer = keras.callbacks.ModelCheckpoint(filepath="xception-less-data-weights.h5",
                                                       verbose=1,
                                                       save_best_only=True)
        self.model.fit_generator(self.next_train_batch(chunk_size=25),
                                 samples_per_epoch=int(self.train_size / (self.num_epochs / 4)),
                                 nb_epoch=self.num_epochs,
                                 validation_data=self.next_validation_batch(chunk_size=20),
                                 nb_val_samples=int(self.validation_size / (self.num_epochs / 4)),
                                 callbacks=[checkpointer, tensorboard],
                                 initial_epoch=start_epoch)
        # should make a callback that writes an epoch number to a file for resuming

neural_net = NeuralNetwork('xception-less-data-weights.h5', continue_training=True, start_epoch=10)
