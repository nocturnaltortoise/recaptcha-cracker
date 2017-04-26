'''
Functions for handling labels, filepaths and images.
'''

import os.path
import glob
from PIL import Image
from keras.utils import np_utils
import inflection
import numpy as np
import re
from config import config


class FilepathPreprocessor:

    @staticmethod
    def create_labels(train_path):
        paths = glob.glob("{0}/*".format(train_path))
        labels_file_path = config['labels_path']
        with open(labels_file_path, 'w+') as labels_file:
            for i, path in enumerate(paths):
                for path_tuple in os.walk(path):
                    root = path_tuple[0]
                    files = path_tuple[2]

                    if len(files) != 0:
                        label = os.path.relpath(root, start=train_path)
                        label_number = i
                        image_files = []
                        for filename in files:
                            path, ext = os.path.splitext(filename)
                            if ext == '.png' or ext == '.jpg' or ext == '.ppm':
                                image_size = "_{0}".format(config['image_size'])
                                if image_size not in path:
                                    image_files.append(os.path.join(label, filename))

                        for file in image_files:
                            file_line = file + " " + str(i) + "\n"
                            labels_file.write(file_line)

    @staticmethod
    def process_filepaths(paths, root_paths):
        new_paths = []
        for path in paths:
            if path[:1] == '/':
                path = path[1:]

            if "_val_" in path:
                full_path = os.path.join(root_paths[1], path)
            else:
                full_path = os.path.join(root_paths[0], path)

            full_path = full_path.replace("\\", "/")
            if os.path.isfile(full_path) and os.path.getsize(full_path) > 0:
                new_paths.append(full_path)
                # filenames for training images have a leading slash,
                # which causes problems on windows with os.path.join

        return new_paths

    @staticmethod
    def change_filepaths_after_resize(paths):
        resize_paths = []
        for path in paths:
            image_size = "_{0}".format(config['image_size'])
            if image_size not in path:
                name, ext = os.path.splitext(path)
                path = name + image_size + ext
            resize_paths.append(path)

        return resize_paths


class ImagePreprocessor:

    @staticmethod
    def resize_images(paths):
        for path in paths:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                filename, ext = os.path.splitext(path)
                image_size = "_{0}".format(config['image_size'])
                new_filename = filename + image_size + ext
                if not os.path.exists(new_filename) or os.path.getsize(new_filename) == 0:
                    # if the file hasn't been resized
                    # or the resized version is corrupt (i.e. zero size)
                    if image_size not in filename:
                        try:
                            image = Image.open(path)
                            image = image.resize(config['image_size_tuple'])
                            image.save(filename + image_size + ext)
                        except OSError:
                            print("OSError caused by file at {0}".format(path))
                            continue
                            # if OSError:
                            # cannot identify image file occurs despite above checks, skip the image

    @staticmethod
    def colour_images(paths):
        for path in paths:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                filename, ext = os.path.splitext(path)
                image_size = "_{0}".format(config['image_size'])
                if image_size in filename:
                    try:
                        image = Image.open(path)
                        if image.mode != "RGB":
                            image = image.convert("RGB")
                            image.save(filename + ext)
                    except OSError:
                        print("OSError caused by file at {0}".format(path))
                        continue
                        # if OSError:
                        # cannot identify image file occurs despite above checks, skip the image

    @staticmethod
    def normalise(image_data):
        image_data = image_data.astype('float')
        image_data /= 255.0
        image_data -= 0.5
        image_data *= 2.0
        return image_data


class LabelProcessor:

    @staticmethod
    def depluralise_string(string):
        singular = inflection.singularize(string)
        return singular

    @staticmethod
    def conflate_labels(image_label):
        conflated_label_words = []
        for word in image_label:
            conflated_label_words.append(LabelProcessor.depluralise_string(word))

        return conflated_label_words

    @staticmethod
    def read_categories(path):
        labels_to_label_names = {}
        with open(path, 'r') as categories_file:
            for line in categories_file:
                label_name, label = line.split(" ")
                label_name = label_name.replace("_", " ")
                label_name = label_name.replace("/", " ")
                labels_to_label_names[int(label)] = label_name

        return labels_to_label_names

    @staticmethod
    def parse_label_from_filename(filename):
        filename_parts = re.findall(r"[a-zA-Z_]+", filename)
        label_parts = []
        for part in filename_parts:
            filetypes = ["jpg", "png", "ppm"]
            if all(file_ext not in part for file_ext in filetypes):
                label_parts.append(part)

        return "/".join(label_parts)

    @staticmethod
    def create_categories_file(labels_file_path):
        filenames, labels = LabelProcessor.read_labels([labels_file_path])
        category_labels = zip(filenames, labels)
        seen_labels = set()
        for category_label in category_labels:
            filename, label = category_label
            if label not in seen_labels:
                filename = LabelProcessor.parse_label_from_filename(filename)
                with open(config['categories_path'], "a") as categories_file:
                    categories_file.write(filename + " " + str(label) + "\n")
                seen_labels.add(label)

    @staticmethod
    def read_labels(paths):
        labels = []
        filenames = []
        for path in paths:
            with open(path, 'r') as label_file:
                for line in label_file:
                    filename, label = line.split(" ")
                    labels.append(int(label.rstrip('\r\n')))
                    filenames.append(filename)

        return filenames, labels

    @staticmethod
    def convert_to_one_hot(labels):
        return np_utils.to_categorical(labels)

    @staticmethod
    def convert_labels_to_label_names(labels):
        chosen_label_names = []
        for image_labels in labels:
            labels_to_label_names = LabelProcessor.read_categories(config['categories_path'])

            label_names = [labels_to_label_names[label] for label in image_labels]

            if not label_names:
                label_names = [""]

            chosen_label_names.append(label_names)

        return chosen_label_names
