import splinter

from captcha_elements import Captcha, Checkbox
from captcha_interaction import CaptchaElement
from exceptions import *

class CaptchaCracker:
    def __init__(self):
        self.captcha_element = None

    def get_new_captcha(self):
        with splinter.Browser() as browser:
            url = 'https://nocturnaltortoise.github.io/captcha'
            browser.visit(url)

            self.captcha_element = CaptchaElement(browser)
            self.captcha_element.click_initial_checkbox()

            self.captcha_element.captcha = Captcha()

            with browser.get_iframe(self.captcha_element.captcha_iframe_name) as iframe:
                self.captcha_element.find_rows_and_cols(iframe)
                self.captcha_element.find_image_url(iframe)
                self.captcha_element.get_image_checkboxes(iframe)
                self.captcha_element.get_captcha_query(iframe)
            print(self.captcha_element.captcha)

captcha_cracker = CaptchaCracker()
captcha_cracker.get_new_captcha()
