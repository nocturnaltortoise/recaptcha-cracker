import time
import splinter
from selenium.common.exceptions import ElementNotVisibleException, StaleElementReferenceException, InvalidElementStateException

from captcha_elements import Captcha, Checkbox
from captcha_interaction import CaptchaElement
from captcha_files import delete_old_images
from exceptions import *
from preprocessors import ImagePreprocessor, LabelProcessor
import nn
import config

class CaptchaCracker:
    def __init__(self):
        self.captcha_element = None
        self.browser = splinter.Browser()
        self.neural_net = nn.NeuralNetwork('weights/xception-less-data-weights.h5')
        self.num_correct = 0
        self.num_guesses = 0

    def setup(self):
        url = config.config['captcha_test_url']
        self.browser.visit(url)
        self.captcha_element = CaptchaElement(self.browser)

    def get_new_captcha(self):
        self.captcha_element.captcha = Captcha()
        captcha = self.captcha_element.captcha

        with self.browser.get_iframe(self.captcha_element.captcha_iframe_name) as iframe:
            self.captcha_element.find_rows_and_cols(iframe)
            self.captcha_element.find_image_url(iframe)
            self.captcha_element.get_captcha_query(iframe)
            self.captcha_element.get_image_checkboxes(iframe)
            self.captcha_element.get_image_urls_for_checkboxes(iframe,
                                                               captcha.checkboxes)
            if self.captcha_changed():
                captcha.checkboxes = [checkbox for checkbox in captcha.checkboxes
                                      if captcha.image_url != checkbox.image_url]
                self.captcha_element.download_new_images()
            else:
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
        return matching_checkboxes

    def reload(self):
        with self.browser.get_iframe(self.captcha_element.captcha_iframe_name) as iframe:
            self.captcha_element.reload(iframe)

    def verify(self):
        with self.browser.get_iframe(self.captcha_element.captcha_iframe_name) as iframe:
            self.captcha_element.verify(iframe)

    def captcha_correct(self):
        if self.captcha_element.captcha_correct():
            return True
        return False

    def print_stats(self):
        print("Guesses: {0}, Correct: {1}, Percent: {2}"
              .format(self.num_guesses,
                      self.num_correct,
                      self.num_correct / captcha_cracker.num_guesses))


def browser_reload():
    delete_old_images()
    captcha_cracker.captcha_element.verify_attempts = 0
    captcha_cracker.captcha_element.browser.reload()
    start()

def start():
    captcha_cracker.setup()
    captcha_cracker.captcha_element.click_initial_checkbox()
    time.sleep(2)
    if captcha_cracker.captcha_correct():
        captcha_cracker.num_guesses += 1
        captcha_cracker.num_correct += 1
        captcha_cracker.print_stats()
        browser_reload()

    while True:
        try:
            captcha_cracker.get_new_captcha()
            captcha_cracker.preprocess_images()
            labels = captcha_cracker.get_predictions()
            matching_checkboxes = captcha_cracker.select_correct_checkboxes(labels)
            verify_attempts = captcha_cracker.captcha_element.verify_attempts
            if matching_checkboxes:
                time.sleep(2)
                if captcha_cracker.captcha_changed():
                    continue
                elif verify_attempts > 0:
                    captcha_cracker.reload()
                else:
                    captcha_cracker.verify()
                captcha_cracker.num_guesses += 1
            else:
                if captcha_cracker.captcha_changed() and verify_attempts < 2:
                    captcha_cracker.verify()
                else:
                    captcha_cracker.reload()
                captcha_cracker.num_guesses += 1

            time.sleep(2)
            if captcha_cracker.captcha_correct():
                captcha_cracker.num_correct += 1

            captcha_cracker.print_stats()

        except SameCaptchaException:
            print("Same CAPTCHA, getting new CAPTCHA.")
            browser_reload()
        except (ElementNotVisibleException,
                StaleElementReferenceException,
                CaptchaImageNotFoundException,
                CheckboxNotFoundException,
                InvalidElementStateException,
                QueryTextNotFoundException) as e:
            print("Crashed: ", e)
            browser_reload()

captcha_cracker = CaptchaCracker()
start()
