'''
Controls input to the reCAPTCHA, using results from the NeuralNetwork class
and preprocessing functions from preprocessors.
'''

import glob
import json
import os
import sys
import time
import uuid

from nltk.corpus import wordnet as wn
from PIL import Image
import requests
from selenium.common.exceptions import StaleElementReferenceException
import splinter

from preprocessors import LabelProcessor, ImagePreprocessor
import nn
from exceptions import CaptchaImageNotFoundException, CheckboxNotFoundException




def write_guesses_to_file(predictions, folder, captcha_text):
    with open('guesses.json','w+') as guess_file:
        existing_predictions = guess_file.read()
        if existing_predictions:
            json_predictions = json.loads(existing_predictions)
            if captcha_text not in json_predictions:
                json_predictions[captcha_text] = {}
            json_predictions[captcha_text][folder] = predictions
            guess_file.write(json.dumps(json_predictions))
        else:
            new_predictions = {}
            new_predictions[captcha_text] = {}
            new_predictions[captcha_text][folder] = predictions
            guess_file.write(json.dumps(new_predictions))


def guess_captcha(browser, neural_net):

    if os.path.exists('logs/current_state.json'):
        with open('logs/current_state.json','r') as current_state_file:
            current_state = json.loads(current_state_file.read())
            correct_score = int(current_state["correct_score"])
            total_guesses = int(current_state["total_guesses"])
    else:
        with open('logs/current_state.json', 'w') as current_state_file:
            new_state = {"correct_score": 0, "total_guesses": 0}
            current_state_file.write(json.dumps(new_state))
            correct_score = 0
            total_guesses = 0

    new_run = True
    while browser.is_element_present_by_css('body > div > div:nth-child(4) > iframe', wait_time=3):
        image_iframe = browser.find_by_css('body > div > div:nth-child(4) > iframe')
        with browser.get_iframe(image_iframe.first['name']) as captcha_iframe:
            # need to keep getting images and image urls until this batch
            # of image urls is the same as the last run
            # i.e. keep selecting images until the captcha stops replacing images
            time.sleep(0.5)
            print("after clicking: ", browser.find_by_css('body > div').first.visible)

            with open('logs/current_state.json', 'r') as current_state_file:
                current_state = json.loads(current_state_file.read())
                correct_score = int(current_state["correct_score"])
                total_guesses = int(current_state["total_guesses"])

            # if new captcha, get checkboxes, download images, pick checkboxes
            if new_run:
                random_folder_name = str(uuid.uuid4())
                picked_checkboxes = None
                # reinitialise picked_checkboxes so previous state doesn't cause problems
                row_count, col_count = find_rows_and_cols(captcha_iframe)

                if row_count == 0 or col_count == 0:
                    break

                print("New CAPTCHA.")
                image_url = find_image_url(captcha_iframe, total_guesses, correct_score)
                captcha_text = get_captcha_query(captcha_iframe)
                captcha_text = LabelProcessor.depluralise_string(captcha_text)
                download_images(image_url, row_count, col_count, captcha_text, random_folder_name)
                ImagePreprocessor.resize_images(glob.glob('*.jpg'))
                ImagePreprocessor.colour_images(glob.glob('*.jpg'))

                labels = neural_net.predict_image_classes()

                predicted_word_labels = LabelProcessor.convert_labels_to_label_names(labels)

                predicted_word_labels = [LabelProcessor.conflate_labels(image_labels) for image_labels in predicted_word_labels]

                image_checkboxes = get_image_checkboxes(row_count, col_count, captcha_iframe)
                picked_checkboxes = pick_checkboxes_matching_query(image_checkboxes, predicted_word_labels, captcha_text)

                write_guesses_to_file(predicted_word_labels, random_folder_name, captcha_text)

                if picked_checkboxes:
                    click_checkboxes(picked_checkboxes)
                    new_run = False
                    new_image_urls = find_image_url(captcha_iframe, total_guesses, correct_score, image_checkboxes)
                else:
                    reload(captcha_iframe, total_guesses, correct_score)
                    new_run = True

            elif any(image_url != new_image_url for new_image_url in new_image_urls):
                print("Some images have changed but CAPTCHA hasn't.")

                image_url = find_image_url(captcha_iframe, total_guesses, correct_score)
                captcha_text = get_captcha_query(captcha_iframe)
                captcha_text = LabelProcessor.depluralise_string(captcha_text)
                download_images(image_url, row_count, col_count, captcha_text, random_folder_name, new_image_urls)
                ImagePreprocessor.resize_images(glob.glob('*.jpg'))
                ImagePreprocessor.colour_images(glob.glob('*.jpg'))

                labels = neural_net.predict_image_classes()
                predicted_word_labels = LabelProcessor.convert_labels_to_label_names(labels)
                predicted_word_labels = [LabelProcessor.conflate_labels(image_labels) for image_labels in predicted_word_labels]
                new_image_checkboxes = get_image_checkboxes(row_count, col_count, captcha_iframe)
                picked_checkboxes = pick_checkboxes_matching_query(new_image_checkboxes, predicted_word_labels, captcha_text)

                if picked_checkboxes:
                    click_checkboxes(picked_checkboxes)
                    new_image_urls = find_image_url(captcha_iframe, total_guesses, correct_score, new_image_checkboxes)
                    new_run = False
                else:
                    verify(captcha_iframe, total_guesses, correct_score)
                    new_run = True
            else:
                print("Not a new captcha and none of the images have changed, verifying.")
                verify(captcha_iframe, total_guesses, correct_score)
                new_run = True

    outer_iframe = browser.find_by_css('body > form > div > div > div > iframe')
    with browser.get_iframe(outer_iframe.first['name']) as iframe:
        checkmarkbox = iframe.find_by_id('recaptcha-anchor')
        if checkmarkbox.has_class('recaptcha-checkbox-checked'):
            browser.reload()
            print("Captchas Correct: {0}".format(correct_score))
            current_state = {'total_guesses': total_guesses, 'correct_score': correct_score+1}
            with open('logs/current_state.json', 'w+') as f:
                f.write(json.dumps(current_state))

            guess_captcha(browser, neural_net)

    if browser.is_element_not_present_by_css('body > form > div > div > div > iframe', wait_time=3):
        print("iframe isn't present and neither is correct checkbox, reloading")
        browser.reload()
        click_initial_checkbox(browser)
        guess_captcha(browser, neural_net)
    elif not browser.find_by_css('body > form > div > div > div > iframe').first.visible:
        print("iframe isn't visible and neither is correct checkbox, reloading")
        browser.reload()
        click_initial_checkbox(browser)
        guess_captcha(browser, neural_net)
    else:
        # print(browser.find_by_css('body > form > div > div > div > iframe').first)
        print("This shouldn't happen")
    # current_state = {'total_guesses': total_guesses, 'correct_score': correct_score}
    # with open('logs/current_state.json', 'a+') as f:
    #     f.write(json.dumps(current_state))


def start_guessing(browser):
    try:
        click_initial_checkbox(browser)
        captcha_window_is_visible = browser.find_by_css('body > div').first.visible
        if captcha_window_is_visible:
            print("Captcha iframe is visible")
            guess_captcha(browser)

    except StaleElementReferenceException:
        print("stale element exception, reloading")
        browser.reload()
        start_guessing(browser)
    except Exception as e:
        browser.reload()
        start_guessing(browser)

with splinter.Browser() as browser:
    URL = "https://nocturnaltortoise.github.io/captcha"
    browser.visit(URL)
    NEURAL_NET = nn.NeuralNetwork('weights/xception-less-data-weights.h5')
    start_guessing(browser)


class CaptchaCracker:
    pass
