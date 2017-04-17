import glob
import json
import os
import sys
from PIL import Image, ImageChops
import numpy as np
import requests
from exceptions import *
from captcha_files import update_state_file, delete_old_images
from captcha_elements import Checkbox
import config
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
        self.verify_attempts = 0

    def reload(self, iframe):
        print("Reloading captcha iframe...")
        if iframe.is_element_present_by_id('recaptcha-reload-button', wait_time=3):
            recaptcha_reload_button = iframe.find_by_id('recaptcha-reload-button')
            CaptchaElement.click_element(recaptcha_reload_button.first)
            self.verify_attempts = 0

            # update_state_file(config.config['state_file_path'], correct=False)

    def captcha_correct(self):
        if self.browser.is_element_present_by_name('undefined', wait_time=3):
            print("found iframe")
            with self.browser.get_iframe('undefined') as iframe:
                if iframe.is_element_present_by_id('recaptcha-anchor', wait_time=3):
                    print("found anchor")
                    recaptcha_checkbox = iframe.find_by_id('recaptcha-anchor')
                    if recaptcha_checkbox.has_class('recaptcha-checkbox-checked'):
                        print("checked")
                        return True
        return False

    @staticmethod
    def join_selectors(selectors):
        return " > ".join(selectors)

    def get_image_urls_for_checkboxes(self, iframe, checkboxes):
        for i, checkbox in enumerate(checkboxes):
            row, col = checkbox.position
            row_col_selector = 'tr:nth-child({0}) > td:nth-child({1})'.format(row, col)
            checkbox_selector = CaptchaElement.join_selectors(
                [self.captcha_table,
                 row_col_selector,
                 self.captcha_image_selector])
            if iframe.is_element_present_by_css(checkbox_selector,
                                                wait_time=3):
                image_url = iframe.find_by_css(checkbox_selector)['src']
                self.captcha.checkboxes[i].image_url = image_url
            else:
                raise CaptchaImageNotFoundException(
                    "Cannot find new image at {0},{1}".format(row, col))

    def find_image_url(self, iframe):
        print("Getting URLs.")
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
        print("Getting image checkbox elements.")
        rows = self.captcha.rows
        cols = self.captcha.cols
        image_checkboxes = []
        for row in range(1, rows+1):
            for col in range(1, cols+1):
                row_col_selector = 'tr:nth-child({0}) > td:nth-child({1})'.format(row, col)
                checkbox_selector = CaptchaElement.join_selectors([self.captcha_table,
                                                                   row_col_selector,
                                                                   'div'])
                if iframe.is_element_present_by_css(checkbox_selector, wait_time=3):
                    checkbox_element = iframe.find_by_css(checkbox_selector)
                    image_checkboxes.append(Checkbox((row, col), checkbox_element))
                else:
                    raise CheckboxNotFoundException("Can't find a checkbox at {0}, {1}"
                                                    .format(row, col))
        self.captcha.checkboxes = image_checkboxes

    def verify(self, iframe):
        print("Clicking verify.")
        if iframe.is_element_present_by_id('recaptcha-verify-button', wait_time=3):
            verify_button = iframe.find_by_id('recaptcha-verify-button')
            CaptchaElement.click_element(verify_button.first)
            self.verify_attempts += 1

    def find_rows_and_cols(self, iframe):
        row_count = 0
        col_count = 0
        captcha_table_selector = '#rc-imageselect-target > table'
        if iframe.is_element_present_by_css(captcha_table_selector,
                                            wait_time=3):
            table = iframe.find_by_css(captcha_table_selector)
            row_count, col_count = table.first['class'].split(" ")[0].split('-')[3]
            row_count, col_count = int(row_count), int(col_count)
            print("rows from find_rows_and_cols: {0}, cols: {1}".format(row_count, col_count))
        self.captcha.rows = row_count
        self.captcha.cols = col_count

    def click_initial_checkbox(self):
        if self.browser.is_element_present_by_name('undefined', wait_time=3):
            with self.browser.get_iframe('undefined') as iframe:
                print("Clicking initial checkbox.")
                checkbox_selector = '#recaptcha-anchor > div.recaptcha-checkbox-checkmark'
                if iframe.is_element_present_by_css(checkbox_selector, wait_time=3):
                    captcha_checkbox = iframe.find_by_css(checkbox_selector)
                    captcha_checkbox.first.click()
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
                print("stale element")
            attempts += 1
        return done

    def click_checkboxes(self, checkboxes):
        for checkbox in checkboxes:
            # print("checkbox visible: ", checkbox.element.visible)
            # if checkbox.element.visible:
            CaptchaElement.click_element(checkbox.element.first)

    def pick_checkboxes_from_positions(self, positions):
        image_checkboxes = self.captcha.checkboxes
        checkboxes = []
        for pos in positions:
            checkboxes.append(image_checkboxes[pos])
        return checkboxes

    def pick_checkboxes_matching_query(self, predicted_word_labels):
        query = self.captcha.query
        matching_labels = set()
        for i, image_labels in enumerate(predicted_word_labels):
            for label in image_labels:
                if " " in label:
                    for word in label.split(" "):
                        if word in query:
                            matching_labels.add(i)
                elif label == query:
                    matching_labels.add(i)
        print(predicted_word_labels)
        print(matching_labels)
        matching_image_checkboxes = self.pick_checkboxes_from_positions(list(matching_labels))
        return matching_image_checkboxes

    def get_captcha_query(self, iframe):
        text_selector = 'div.rc-imageselect-desc-no-canonical > strong'
        if iframe.is_element_present_by_css(text_selector, wait_time=3):
            captcha_text = iframe.find_by_css(text_selector).first['innerHTML']
            self.captcha.query = LabelProcessor.depluralise_string(captcha_text)
            print("query: ", self.captcha.query)
        else:
            raise QueryTextNotFoundException("Can't find query text.")

    def download_new_images(self):
        print("Downloading images.")
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

        # captcha_folder = os.path.join('datasets', 'captchas', self.captcha.query)
        filepaths = []
        images = []
        same_images = []
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
                # this should be somewhere else
                # if not os.path.exists(captcha_folder):
                #     os.mkdir(captcha_folder)
                #
                # if not os.path.exists(captcha_folder + "/" + random_folder_name):
                #     os.mkdir("{0}/{1}".format(captcha_folder, random_folder_name))
                # individual_captcha_image.save("{0}/{1}/{2}-{3}.jpg".format(captcha_folder, random_folder_name, row, col), "JPEG")
