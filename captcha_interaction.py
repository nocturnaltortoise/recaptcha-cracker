import os
import uuid
import random
from exceptions import *
from PIL import Image
import requests
from captcha_files import delete_old_images
from captcha_elements import Checkbox
from preprocessors import LabelProcessor
from selenium.common.exceptions import StaleElementReferenceException


class CaptchaElement:
    '''Provides functions to interact with the reCAPTCHA interface.'''
    def __init__(self, browser):
        self.captcha = None
        self.browser = browser
        self.captcha_iframe_element = browser.find_by_css('body > div > div:nth-child(4) > iframe')
        self.captcha_iframe_name = self.captcha_iframe_element.first['name']
        self.captcha_table = '#rc-imageselect-target > table > tbody'
        self.captcha_image_selector = 'div > div.rc-image-tile-wrapper > img'

    def reload(self, iframe):
        if iframe.is_element_present_by_id('recaptcha-reload-button', wait_time=3):
            recaptcha_reload_button = iframe.find_by_id('recaptcha-reload-button')
            CaptchaElement.click_element(recaptcha_reload_button.first)

    def captcha_correct(self):
        if self.browser.is_element_present_by_name('undefined', wait_time=3):
            with self.browser.get_iframe('undefined') as iframe:
                if iframe.is_element_present_by_id('recaptcha-anchor', wait_time=3):
                    recaptcha_checkbox = iframe.find_by_id('recaptcha-anchor')
                    if recaptcha_checkbox.has_class('recaptcha-checkbox-checked'):
                        return True
        return False

    @staticmethod
    def join_selectors(selectors):
        return " > ".join(selectors)

    def find_image_url(self, iframe):
        # get the first image in the grid, which has the url of the main image
        # if this code is called after the first image has changed it will
        # produce unexpected results
        row_col_selector = 'tr:nth-child(1) > td:nth-child(1)'
        image_selector = CaptchaElement.join_selectors([self.captcha_table,
                                                        row_col_selector,
                                                        self.captcha_image_selector])
        if iframe.is_element_present_by_css(image_selector, wait_time=3):
            image_url = iframe.find_by_css(image_selector)['src']
            self.captcha.image_url = image_url
        else:
            raise CaptchaImageNotFoundException("Cannot find original image.")

    def get_image_checkboxes(self, iframe):
        rows = self.captcha.rows
        cols = self.captcha.cols
        image_checkboxes = []
        for row in range(1, rows+1):
            for col in range(1, cols+1):
                row_col_selector = 'tr:nth-child({0}) > td:nth-child({1})'.format(row, col)
                checkbox_selector = CaptchaElement.join_selectors([self.captcha_table,
                                                                   row_col_selector,
                                                                   'div'])
                image_selector = CaptchaElement.join_selectors([self.captcha_table,
                                                                row_col_selector,
                                                                self.captcha_image_selector])
                if iframe.is_element_present_by_css(checkbox_selector, wait_time=3):
                    checkbox_element = iframe.find_by_css(checkbox_selector)
                    image_element = iframe.find_by_css(image_selector)
                    image_url = image_element['src']
                    image_checkboxes.append(Checkbox((row, col), checkbox_element, image_url))
                else:
                    raise CheckboxNotFoundException("Can't find a checkbox at {0}, {1}"
                                                    .format(row, col))
        self.captcha.checkboxes = image_checkboxes

    def verify(self, iframe):
        if iframe.is_element_present_by_id('recaptcha-verify-button', wait_time=3):
            verify_button = iframe.find_by_id('recaptcha-verify-button')
            CaptchaElement.click_element(verify_button.first)

    def find_rows_and_cols(self, iframe):
        row_count = 0
        col_count = 0
        captcha_table_selector = '#rc-imageselect-target > table'
        if iframe.is_element_present_by_css(captcha_table_selector,
                                            wait_time=3):
            table = iframe.find_by_css(captcha_table_selector)
            row_count, col_count = table.first['class'].split(" ")[0].split('-')[3]
            row_count, col_count = int(row_count), int(col_count)
        self.captcha.rows = row_count
        self.captcha.cols = col_count

    def click_initial_checkbox(self):
        if self.browser.is_element_present_by_name('undefined', wait_time=3):
            with self.browser.get_iframe('undefined') as iframe:
                checkbox_selector = '#recaptcha-anchor > div.recaptcha-checkbox-checkmark'
                if iframe.is_element_present_by_css(checkbox_selector, wait_time=3):
                    captcha_checkbox = iframe.find_by_css(checkbox_selector)
                    CaptchaElement.click_element(captcha_checkbox.first)
                else:
                    raise InitialCheckboxNotFoundException("Can't find initial checkbox.")
        else:
            raise IFrameNotFoundException("Can't find initial iframe.")

    @staticmethod
    def click_element(element):
        done = False
        attempts = 0
        while attempts < 3:
            try:
                element.click()
                done = True
                break
            except StaleElementReferenceException:
                pass
            attempts += 1
        return done

    def click_checkboxes(self, checkboxes):
        for checkbox in checkboxes:
            CaptchaElement.click_element(checkbox.element.first)

    def pick_checkboxes_matching_query(self):
        query = self.captcha.query
        matching_checkboxes = set()
        for checkbox in self.captcha.checkboxes:
            image_labels = checkbox.predictions
            for label in image_labels:
                if " " in label:
                    for word in label.split(" "):
                        if word in query:
                            matching_checkboxes.add(checkbox)
                elif label == query:
                    matching_checkboxes.add(checkbox)
        return matching_checkboxes

    def pick_random_checkboxes(self):
        checkboxes = self.captcha.checkboxes
        num_to_pick = random.randint(0, len(checkboxes))
        if num_to_pick != 0:
            picked_checkboxes = random.sample(checkboxes, num_to_pick)
        else:
            picked_checkboxes = []
        return picked_checkboxes

    def get_captcha_query(self, iframe):
        text_selector = 'div.rc-imageselect-desc-no-canonical > strong'
        if iframe.is_element_present_by_css(text_selector, wait_time=3):
            captcha_text = iframe.find_by_css(text_selector).first['innerHTML']
            self.captcha.query = LabelProcessor.depluralise_string(captcha_text)
        else:
            raise QueryTextNotFoundException("Can't find query text.")

    def download_new_images(self):
        delete_old_images()

        for i, checkbox in enumerate(self.captcha.checkboxes):
            url = checkbox.image_url
            img = Image.open(requests.get(url, stream=True).raw)
            filepath = "new-captcha-{0}.jpg".format(i)
            img.save(filepath, "JPEG")
            checkbox.image_path = filepath

    def download_initial_image(self):
        img = Image.open(requests.get(self.captcha.image_url, stream=True).raw)
        img.save("original-captcha-image.jpg", "JPEG")

        width = img.size[0] / self.captcha.cols
        height = img.size[1] / self.captcha.rows

        filepaths = []
        images = []
        for row in range(self.captcha.rows):
            for col in range(self.captcha.cols):
                dimensions = (col * width, row * height, col * width + width, row * height + height)
                individual_captcha_image = img.crop(dimensions)
                filepath = "captcha-{0}-{1}.jpg".format(row, col)
                images.append((individual_captcha_image, row, col))

        delete_old_images()

        for image in images:
            captcha_image, row, col = image
            filepath = "captcha-{0}-{1}.jpg".format(row, col)
            captcha_image.save(filepath, "JPEG")
            filepaths.append(filepath)

        for i, checkbox in enumerate(self.captcha.checkboxes):
            checkbox.image_path = filepaths[i]

        self.save_images_permanently(self.captcha.checkboxes, images)

    def save_images_permanently(self, checkboxes, images):
        captcha_folder = os.path.join('datasets', 'captchas', self.captcha.query)

        if not os.path.exists(captcha_folder):
            os.makedirs(captcha_folder)

        random_folder_name = str(uuid.uuid4())
        self.captcha.random_id = random_folder_name

        for i, image in enumerate(images):
            image, row, col = image
            checkbox = checkboxes[i]

            random_folder_path = os.path.join(captcha_folder, random_folder_name)
            if not os.path.exists(random_folder_path):
                os.mkdir(random_folder_path)

            filepath = os.path.join(random_folder_path, "{0}-{1}.jpg".format(row, col))
            image.save(filepath, "JPEG")

            checkbox.permanent_path = filepath
