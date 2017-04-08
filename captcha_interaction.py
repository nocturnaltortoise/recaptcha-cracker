import glob
import json
import os
import sys
from exceptions import *
from captcha_files import update_state_file, delete_old_images
from captcha_elements import Checkbox


class CaptchaElement:
    '''Provides functions to interact with the reCAPTCHA interface.'''
    def __init__(self, browser):
        self.captcha = None
        self.browser = browser
        self.captcha_iframe_element = browser.find_by_css('body > div > div:nth-child(4) > iframe')
        self.captcha_iframe = browser.get_iframe(self.captcha_iframe_element.first['name'])
        self.captcha_table = '#rc-imageselect-target > table > tbody'
        self.captcha_image_selector = 'div > div.rc-image-tile-wrapper > img'
        self.state_file_path = 'logs/current_state.json'

    def reload(self):
        print("Reloading captcha iframe...")
        if self.captcha_iframe.is_element_present_by_id('recaptcha-reload-button', wait_time=3):
            recaptcha_reload_button = self.captcha_iframe.find_by_id('recaptcha-reload-button')
            recaptcha_reload_button.first.click()

            update_state_file(self.state_file_path, correct=False)

    @staticmethod
    def join_selectors(selectors):
        return " > ".join(selectors)

    def find_new_image_urls(self, changed_images):
        try:
            image_urls = []
            for image in changed_images:
                row, col = image['position']
                row_col_selector = 'tr:nth-child({0}) > td:nth-child({1})'.format(row, col)
                changed_image_selector = CaptchaElement.join_selectors(
                    [self.captcha_table,
                     row_col_selector,
                     self.captcha_image_selector])
                if self.captcha_iframe.is_element_present_by_css(changed_image_selector,
                                                                 wait_time=3):
                    image_url = self.captcha_iframe.find_by_css(changed_image_selector)['src']
                    image_urls.append(image_url)
                else:
                    raise CaptchaImageNotFoundException(
                        "Cannot find new image at {0},{1}".format(row, col))
            self.captcha.changed_urls = image_urls
        except CaptchaImageNotFoundException as image_not_found:
            print(image_not_found.message, file=sys.stderr)
            self.reload()

    def find_image_url(self):
        print("Getting URLs.")
        try:
            # get the first image in the grid, which has the url of the main image
            # if this code is called after the first image has changed it will
            # produce unexpected results
            row_col_selector = 'tr:nth-child(1) > td:nth-child(1)'
            image_selector = CaptchaElement.join_selectors([self.captcha_table,
                                                            row_col_selector,
                                                            self.captcha_image_selector])
            if self.captcha_iframe.is_element_present_by_css(image_selector, wait_time=3):
                image_url = self.captcha_iframe.find_by_css(image_selector)['src']
                self.captcha.image_url = image_url
            else:
                raise CaptchaImageNotFoundException("Cannot find original image.")
        except CaptchaImageNotFoundException as image_not_found:
            print(image_not_found.message, file=sys.stderr)
            self.reload()

    def get_image_checkboxes(self):
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

                if self.captcha_iframe.is_element_present_by_css(checkbox_selector, wait_time=3):
                    checkbox_element = self.captcha_iframe.find_by_css(checkbox_selector)
                    image_checkboxes.append(Checkbox((row, col), checkbox_element))
                else:
                    raise CheckboxNotFoundException("Can't find a checkbox at {0}, {1}"
                                                    .format(row, col))
        self.captcha.checkboxes = image_checkboxes

    def verify(self):
        print("Clicking verify.")
        if self.captcha_iframe.is_element_present_by_id('recaptcha-verify-button', wait_time=3):
            verify_button = self.captcha_iframe.find_by_id('recaptcha-verify-button')
            verify_button.first.click()

    def find_rows_and_cols(self):
        row_count = 0
        col_count = 0
        captcha_table_selector = '#rc-imageselect-target > table'
        if self.captcha_iframe.is_element_present_by_css(captcha_table_selector,
                                                         wait_time=3):
            table = self.captcha_iframe.find_by_css(captcha_table_selector)
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

    def click_checkboxes(self):
        checkboxes = self.captcha.checkboxes
        if checkboxes:
            for checkbox in checkboxes:
                if checkbox['element'].visible:
                    checkbox['element'].click()

    def pick_checkboxes_from_positions(self, positions):
        image_checkboxes = self.captcha.checkboxes
        checkboxes = []
        for pos in positions:
            checkboxes.append(image_checkboxes[pos])

        return checkboxes

    def pick_checkboxes_matching_query(self, predicted_word_labels):
        query = self.captcha.query

        matching_labels = []
        for i, image_labels in enumerate(predicted_word_labels):
            for label in image_labels:
                if " " in label:
                    for word in label.split(" "):
                        if word == query:
                            matching_labels.append(i)
                elif label == query:
                    matching_labels.append(i)

        matching_image_checkboxes = self.pick_checkboxes_from_positions(matching_labels)
        return matching_image_checkboxes

    def get_captcha_query(self):
        text_selector = 'div.rc-imageselect-desc-no-canonical'
        if self.captcha_iframe.is_element_present_by_css(text_selector, wait_time=3):
            captcha_text = self.captcha_iframe.find_by_css(text_selector).first['innerHTML']
            self.captcha.query = captcha_text

    def download_images(self, random_folder_name):
        changed_urls = self.captcha.changed_urls
        # this function does too much - too many args and too many local vars
        print("Downloading images.")
        if changed_urls:
            delete_old_images()

            for i, url in enumerate():
                img = Image.open(requests.get(url, stream=True).raw)
                img.save("new-captcha-{0}.jpg".format(i), "JPEG")
        else:
            img = Image.open(requests.get(image_url, stream=True).raw)
            img.save("original-captcha-image.jpg", "JPEG")

            delete_old_images()

            width = img.size[0] / col_count
            height = img.size[1] / row_count

            captcha_folder = "datasets/captchas/{0}".format(captcha_text)
            for row in range(row_count):
                for col in range(col_count):
                    dimensions = (col * width, row * height, col * width + width, row * height + height)
                    individual_captcha_image = img.crop(dimensions)
                    individual_captcha_image.save("captcha-{0}-{1}.jpg".format(row, col), "JPEG")

                    if not os.path.exists(captcha_folder):
                        os.mkdir(captcha_folder)

                    if not os.path.exists(captcha_folder + "/" + random_folder_name):
                        os.mkdir("{0}/{1}".format(captcha_folder, random_folder_name))
                    individual_captcha_image.save("{0}/{1}/{2}-{3}.jpg".format(captcha_folder, random_folder_name, row, col), "JPEG")
