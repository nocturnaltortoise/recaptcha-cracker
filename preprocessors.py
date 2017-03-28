import os.path
from PIL import Image
from keras.utils import np_utils
import glob
import inflection


class FilepathPreprocessor:

    @staticmethod
    def create_labels(train_path):
        paths = glob.glob("{0}/*".format(train_path))
        for i, path in enumerate(paths):
            for path_tuple in list(os.walk(path)):
                root = path_tuple[0]
                dirs = path_tuple[1]
                files = path_tuple[2]

                if len(files) != 0:
                    label = root.replace("E:\datasets\captcha-dataset\\","")
                    label_number = i
                    files = [filename for filename in files if "_110x110" not in filename]
                    files = [os.path.join(label, filename) for filename in files]
                    print(label_number, len(files))
            # thing = FilepathPreprocessor.walk_tree(path)
            # print(type(thing))
            # next_dir = next(os.walk(path))
            # subpaths = next_dir[1]
            #
            # if len(next_dir) == 2:
            #     for subpath in subpaths:
            #         path = os.path.join(path, subpath)
            #         next_dir = next(os.walk(subpath))
            #         files = next_dir[2]
            # else:
            #     files = next_dir[2]
            #
            # print(path, len(files))

    @staticmethod
    def walk_tree(path):
        root, dirs, files = next(os.walk(path))
        if dirs:
            for dirname in dirs:
                print(os.path.join(root, dirname))
                FilepathPreprocessor.walk_tree(os.path.join(root, dirname))
        else:
            return root, dirs, files

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
            else:
                print(full_path)
                # filenames for training images have a leading slash,
                # which causes problems on windows with os.path.join

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
    def depluralise_string(string):
        singular = inflection.singularize(string)
        return singular

    @staticmethod
    def conflate_labels(image_label):
        print("word labels: ", image_label)
        conflated_labels = []
        for word in image_label:
            conflated_labels.append(LabelProcessor.depluralise_string(word))

        return conflated_labels

    @staticmethod
    def read_categories(path):
        labels_to_label_names = {}
        with open(path, 'r') as f:
            for line in f:
                label_name, label = line.split(" ")
                label_name = label_name[3:]  # get rid of the folder name and slashes
                label_name = label_name.replace("_", " ")
                label_name = label_name.replace("/", " ")
                labels_to_label_names[int(label)] = label_name

        return labels_to_label_names

    @staticmethod
    def read_labels(paths):
        labels = []
        filenames = []
        for path in paths:
            print("reading labels in {0}".format(path))
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
        print(labels)
        chosen_label_names = []
        for image_labels in labels:
            # names = np.load('names.npy')
            labels_to_label_names = LabelProcessor.read_categories('categories_places365.txt')

            label_names = [labels_to_label_names[label] for label in image_labels]

            if not label_names:
                label_names = [""]

            chosen_label_names.append(label_names)

        return chosen_label_names

    @staticmethod
    def create_label_file_from_files(paths):
        with open('E:\datasets\extra_data_labels.txt', 'w+') as f:
            for path in paths:
                for filepath in glob.glob(path):
                    print(filepath)
                    if "_32x32" not in filepath and "_110x110" not in filepath:
                        if "trafficsigns-train" in filepath and "/00014" not in filepath:
                            if ".ppm" in filepath:
                                f.write(filepath + " " + "366" + "\n")
                        elif "svhn-train" in filepath:
                            if ".png" in filepath:
                                f.write(filepath + " " + "365" + "\n")
                        else:
                            if ".ppm" in filepath:
                                f.write(filepath + " " + "367" + "\n")

        # maybe take a labels - foldername dictionary
        # e.g. svhn-train: 365, traffic-signs-train: 366, traffic-signs-train/00014: 367
