import time
import splinter

from captcha_elements import Captcha, Checkbox
from captcha_interaction import CaptchaElement
from exceptions import *
from preprocessors import ImagePreprocessor, LabelProcessor
import nn
import config

class CaptchaCracker:
    def __init__(self):
        self.captcha_element = None
        self.browser = splinter.Browser()
        self.neural_net = None

    def setup(self):
        url = config.config['captcha_test_url']
        self.browser.visit(url)
        self.neural_net = nn.NeuralNetwork('weights/xception-less-data-weights.h5')

    def get_new_captcha(self):
        self.captcha_element = CaptchaElement(self.browser)
        self.captcha_element.click_initial_checkbox()

        self.captcha_element.captcha = Captcha()
        captcha = self.captcha_element.captcha

        with self.browser.get_iframe(self.captcha_element.captcha_iframe_name) as iframe:
            self.captcha_element.find_rows_and_cols(iframe)
            self.captcha_element.find_image_url(iframe)
            self.captcha_element.get_captcha_query(iframe)
            self.captcha_element.get_image_checkboxes(iframe)
            self.captcha_element.get_image_urls_for_checkboxes(iframe,
                                                               captcha.checkboxes)
            self.captcha_element.download_initial_image()

    def captcha_changed(self):
        captcha = self.captcha_element.captcha
        return any(captcha.image_url != checkbox.image_url for checkbox in captcha.checkboxes)

    def preprocess_images(self):
        image_paths = [checkbox.image_path for checkbox in self.captcha_element.captcha.checkboxes]
        ImagePreprocessor.resize_images(image_paths)
        ImagePreprocessor.colour_images(image_paths)

    def get_predictions(self):
        all_labels = self.neural_net.predict_image_classes()
        all_label_names = LabelProcessor.convert_labels_to_label_names(all_labels)
        all_label_names = [LabelProcessor.conflate_labels(label_names) for label_names in all_label_names]
        print("labels: ", all_label_names)
        return all_label_names

    def select_correct_checkboxes(self, labels):
        matching_checkboxes = self.captcha_element.pick_checkboxes_matching_query(labels)
        with self.browser.get_iframe(self.captcha_element.captcha_iframe_name) as iframe:
            self.captcha_element.click_checkboxes(matching_checkboxes)

captcha_cracker = CaptchaCracker()
captcha_cracker.setup()
captcha_cracker.get_new_captcha()
captcha_cracker.preprocess_images()
labels = captcha_cracker.get_predictions()
captcha_cracker.select_correct_checkboxes(labels)
time.sleep(10)
