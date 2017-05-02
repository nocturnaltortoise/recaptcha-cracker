import time
import splinter
from selenium.common.exceptions import ElementNotVisibleException, StaleElementReferenceException, InvalidElementStateException

from captcha_elements import Captcha, Checkbox
from captcha_interaction import CaptchaElement
from captcha_files import delete_old_images, write_guesses_to_file
from exceptions import *
from preprocessors import ImagePreprocessor, LabelProcessor
import nn
from config import config

class CaptchaCracker:
    def __init__(self):
        self.captcha_element = None
        self.browser = splinter.Browser()
        self.neural_net = nn.NeuralNetwork(weights_file=config['weights_path'])
        self.num_correct = 0
        self.num_guesses = 0
        self.old_captcha_urls = []

    def setup(self):
        url = config['captcha_test_url']
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
            checkbox_urls = [checkbox.image_url for checkbox in captcha.checkboxes]
            if checkbox_urls == self.old_captcha_urls:
                raise SameCaptchaException("Same CAPTCHA as previous CAPTCHA.")
            else:
                self.old_captcha_urls = checkbox_urls

            if self.captcha_changed():
                captcha.checkboxes = [checkbox for checkbox in captcha.checkboxes
                                      if captcha.image_url != checkbox.image_url]
                self.captcha_element.download_new_images()
            else:
                self.captcha_element.download_initial_image()

    def captcha_changed(self):
        captcha = self.captcha_element.captcha
        return any(captcha.image_url != checkbox.image_url for checkbox in captcha.checkboxes)

    def refresh_checkboxes(self):
        with self.browser.get_iframe(self.captcha_element.captcha_iframe_name) as iframe:
            self.captcha_element.get_image_checkboxes(iframe)

    def preprocess_images(self):
        image_paths = [checkbox.image_path for checkbox in self.captcha_element.captcha.checkboxes]
        ImagePreprocessor.resize_images(image_paths)
        ImagePreprocessor.colour_images(image_paths)

    def get_predictions(self):
        all_labels = self.neural_net.predict_image_classes(self.captcha_element.captcha.checkboxes)
        all_label_names = LabelProcessor.convert_labels_to_label_names(all_labels)
        all_label_names = [LabelProcessor.conflate_labels(label_names) for label_names in all_label_names]
        for i, checkbox in enumerate(self.captcha_element.captcha.checkboxes):
            checkbox.predictions = all_label_names[i]

    def select_correct_checkboxes(self):
        matching_checkboxes = self.captcha_element.pick_checkboxes_matching_query()
        with self.browser.get_iframe(self.captcha_element.captcha_iframe_name) as iframe:
            self.captcha_element.click_checkboxes(matching_checkboxes)
        return matching_checkboxes

    def select_random_checkboxes(self):
        random_checkboxes = self.captcha_element.pick_random_checkboxes()
        with self.browser.get_iframe(self.captcha_element.captcha_iframe_name) as iframe:
            self.captcha_element.click_checkboxes(random_checkboxes)
        return random_checkboxes

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
        print("Max: {0}, Guesses: {1}, Correct: {2}, Percent: {3}"
              .format(MAX_RUNS,
                      self.num_guesses,
                      self.num_correct,
                      self.num_correct / captcha_cracker.num_guesses))


def browser_reload():
    captcha_cracker.captcha_element.verify_attempts = 0
    captcha_cracker.captcha_element.browser.reload()
    start()

MAX_RUNS = 1000
def start():
    captcha_cracker.setup()
    captcha_cracker.captcha_element.click_initial_checkbox()
    time.sleep(0.5)
    if captcha_cracker.captcha_correct():
        captcha_cracker.num_guesses += 1
        captcha_cracker.num_correct += 1

        captcha_cracker.print_stats()
        browser_reload()

    while captcha_cracker.num_guesses < MAX_RUNS:
        try:
            captcha_cracker.get_new_captcha()
            captcha_cracker.preprocess_images()
            captcha_cracker.get_predictions()
            # matching_checkboxes = captcha_cracker.select_random_checkboxes()
            # uncomment to run random clicker, 
            # comment get_predictions and select_correct_checkboxes as well if doing that
            matching_checkboxes = captcha_cracker.select_correct_checkboxes()
            time.sleep(1)
            # refresh the checkboxes as the urls may have changed, and we need to check the new urls
            captcha_cracker.refresh_checkboxes()

            if matching_checkboxes:
                if captcha_cracker.captcha_changed():
                    continue
                else:
                    captcha_cracker.verify()
                    captcha_cracker.num_guesses += 1
            else:
                if captcha_cracker.captcha_changed():
                    captcha_cracker.verify()
                else:
                    captcha_cracker.reload()
                captcha_cracker.num_guesses += 1

            time.sleep(0.5)
            captcha_correct = captcha_cracker.captcha_correct()
            if captcha_correct:
                captcha_cracker.num_correct += 1

            write_guesses_to_file(captcha_cracker.captcha_element.captcha, matching_checkboxes, captcha_correct)
            captcha_cracker.print_stats()

        except SameCaptchaException:
            browser_reload()
        except (ElementNotVisibleException,
                StaleElementReferenceException,
                CaptchaImageNotFoundException,
                CheckboxNotFoundException,
                InvalidElementStateException,
                QueryTextNotFoundException):
            browser_reload()

captcha_cracker = CaptchaCracker()
start()
