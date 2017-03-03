import os.path
from PIL import Image
import numpy as np
from keras.utils import np_utils


class FilepathPreprocessor:

    @staticmethod
    def process_filepaths(paths, root_path, remove_leading_slash=True):
        new_paths = []
        if remove_leading_slash:
            for path in paths:
                full_path = os.path.join(root_path, path[1:])
                if os.path.isfile(full_path) and os.path.getsize(full_path) > 0:
                    new_paths.append(full_path)
                    # filenames for training images have a leading slash,
                    # which causes problems on windows with os.path.join
        else:
            for path in paths:
                full_path = os.path.join(root_path, path)
                if os.path.isfile(full_path) and os.path.getsize(full_path) > 0:
                    new_paths.append(full_path)

        return new_paths

    @staticmethod
    def change_filepaths_after_resize(paths):
        resize_paths = []
        for path in paths:
            if "_110x110" not in path:
                name, ext = os.path.splitext(path)
                path = name + "_110x110" + ext
            resize_paths.append(path)

        return resize_paths


class ImagePreprocessor:

    @staticmethod
    def resize_images(paths):
        print("resizing images")
        for path in paths:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                filename, ext = os.path.splitext(path)
                if not os.path.exists(filename + "_110x110" + ext) or os.path.getsize(filename + "_110x110" + ext) == 0:
                    # if the file hasn't been resized or the resized version is corrupt (i.e. zero size)
                    if "_110x110" not in filename:
                        try:
                            image = Image.open(path)
                            image = image.resize((110, 110))
                            image.save(filename + "_110x110" + ext)
                        except OSError:
                            print("OSError caused by file at {0}".format(path))
                            continue
                            # if OSError: cannot identify image file occurs despite above checks, skip the image

    @staticmethod
    def colour_images(paths):
        print("colouring images")
        for path in paths:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                filename, ext = os.path.splitext(path)
                if "_110x110" in filename:
                    try:
                        image = Image.open(path)
                        if image.mode != "RGB":
                            print("image not RGB, colouring")
                            image = image.convert("RGB")
                            image.save(filename + ext)
                    except OSError:
                        print("OSError caused by file at {0}".format(path))
                        continue  # if OSError: cannot identify image file occurs despite above checks, skip the image

    @staticmethod
    def normalise(image_data):
        image_data = image_data.astype('float')
        image_data /= 255.0
        image_data -= 0.5
        image_data *= 2.0
        return image_data


class LabelProcessor:

    @staticmethod
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

    @staticmethod
    def convert_to_one_hot(labels):
        print("converting to one hot")
        return np_utils.to_categorical(labels)

    @staticmethod
    def convert_labels_to_label_names(labels):
        chosen_label_names = []
        for image_labels in labels:
            # names = np.load('names.npy')

            # label_names = [names[label] for label in image_labels]

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