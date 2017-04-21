from splinter import Browser
import os.path
import json
import time


def reload(captcha_iframe):
    time.sleep(1)
    if captcha_iframe.is_element_present_by_id('recaptcha-reload-button', wait_time=3):
        recaptcha_reload_button = captcha_iframe.find_by_id('recaptcha-reload-button')
        recaptcha_reload_button.first.click()


def record_captcha_question(captcha_iframe, queries):
    text_xpath = '//*[@id="rc-imageselect"]/div[2]/div[1]/div[1]/div[1]/strong'
    if captcha_iframe.is_element_present_by_xpath(text_xpath, wait_time=3):
        captcha_text = captcha_iframe.find_by_xpath(text_xpath).first['innerHTML']

        if captcha_text in queries:
            queries[captcha_text] += 1
        else:
            queries[captcha_text] = 1

    return queries


def save_queries(queries):
    if queries != {}:
        print(queries)
        with open('queries.json', 'w+') as f:
            json.dump(queries, f)


def make_mistake(captcha_iframe, mistake_count):
    if mistake_count == 0:
        if captcha_iframe.is_element_present_by_id('recaptcha-verify-button', wait_time=3):
            verify_button = captcha_iframe.find_by_id('recaptcha-verify-button')
            verify_button.first.click()


def load_queries(path):
    queries = None
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        with open(path, 'r') as f:
            queries = json.load(f)

    return queries


with Browser() as browser:
    url = "https://nocturnaltortoise.github.io/captcha"
    browser.visit(url)

    queries = load_queries('queries.json')
    if not queries:
        queries = {}

    try:
        with browser.get_iframe('undefined') as iframe:
            if iframe.is_element_present_by_xpath('//div[@class="recaptcha-checkbox-checkmark"]', wait_time=3):
                captcha_checkbox = iframe.find_by_xpath('//div[@class="recaptcha-checkbox-checkmark"]')
                captcha_checkbox.first.click()

        mistake_count = 0
        while True:
            if browser.is_element_present_by_css('body > div > div:nth-child(4) > iframe', wait_time=3):
                captcha_iframe_name = browser.find_by_css('body > div > div:nth-child(4) > iframe').first['name']

                with browser.get_iframe(captcha_iframe_name) as captcha_iframe:
                    make_mistake(captcha_iframe, mistake_count)
                    mistake_count += 1
                    queries = record_captcha_question(captcha_iframe, queries)
                    reload(captcha_iframe)
                    save_queries(queries)

    except Exception as e:
        print(e)
