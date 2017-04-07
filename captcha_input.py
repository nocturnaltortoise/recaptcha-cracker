'''
Controls input to the reCAPTCHA, using results from the NeuralNetwork class
and preprocessing functions from preprocessors.
'''

import glob
import json
import os
import time
import uuid

from nltk.corpus import wordnet as wn
from PIL import Image
import requests
from selenium.common.exceptions import StaleElementReferenceException
import splinter

from preprocessors import LabelProcessor, ImagePreprocessor
import nn


def find_image_url(captcha_iframe, total_guesses, correct_score, image_checkboxes=None):
    print("Getting URLs.")

    if image_checkboxes:
        image_urls = []
        for checkbox in image_checkboxes:
            row, col = checkbox['position']
            changed_image_xpath = '//*[@id="rc-imageselect-target"]/table/tbody/tr[{0}]/td[{1}]/div/div[1]/img'\
                .format(row, col)
            if captcha_iframe.is_element_present_by_xpath(changed_image_xpath, wait_time=3):
                image_url = captcha_iframe.find_by_xpath(changed_image_xpath)['src']
                image_urls.append(image_url)
            else:
                print("can't find image")
        return image_urls
    else:
        image_xpath = '//*[@id="rc-imageselect-target"]/table/tbody/tr[1]/td[1]/div/div[1]/img'
        if captcha_iframe.is_element_present_by_xpath(image_xpath, wait_time=3):
            image_url = captcha_iframe.find_by_xpath(image_xpath)['src']
        else:
            print("can't find image")
            reload(captcha_iframe, total_guesses, correct_score)
        return image_url


def pick_checkboxes_from_positions(positions, image_checkboxes):
    checkboxes = []
    for pos in positions:
        checkboxes.append(image_checkboxes[pos])

    return checkboxes


def get_image_checkboxes(rows, cols, captcha_iframe):
    print("Getting image checkbox elements.")
    image_checkboxes = []
    for i in range(1, rows+1):
        for j in range(1, cols+1):
            checkbox_xpath = '//*[@id="rc-imageselect-target"]/table/tbody/tr[{0}]/td[{1}]/div'\
                                .format(i, j)

            if captcha_iframe.is_element_present_by_xpath(checkbox_xpath, wait_time=3):
                image_checkboxes.append({
                    'checkbox': captcha_iframe.find_by_xpath(checkbox_xpath),
                    'position': (i, j)
                })
            else:
                print("Can't find a checkbox at {0}, {1}".format(i, j))
    return image_checkboxes


def verify(captcha_iframe, total_guesses, correct_score):
    print("Clicking verify.")
    if captcha_iframe.is_element_present_by_id('recaptcha-verify-button', wait_time=3):
        verify_button = captcha_iframe.find_by_id('recaptcha-verify-button')
        verify_button.first.click()
        current_state = {'total_guesses': total_guesses+1, 'correct_score': correct_score}
        with open('logs/current_state.json', 'w+') as current_state_file:
            current_state_file.write(json.dumps(current_state))


def delete_old_images():
    old_captcha_images = glob.glob('*captcha-*.jpg')
    for image in old_captcha_images:
        os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), image))


def download_images(image_url, row_count, col_count, captcha_text, random_folder_name, image_urls=None):
    # this function does too much - too many args and too many local vars
    print("Downloading images.")
    if image_urls:
        delete_old_images()

        for i, url in enumerate(image_urls):
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


def find_rows_and_cols(captcha_iframe):
    row_count = 0
    col_count = 0
    if captcha_iframe.is_element_present_by_css('#rc-imageselect-target > table', wait_time=3):
        table = captcha_iframe.find_by_css('#rc-imageselect-target > table')
        row_count, col_count = table.first['class'].split(" ")[0].split('-')[3]
        row_count, col_count = int(row_count), int(col_count)
        print("rows from find_rows_and_cols: {0}, cols: {1}".format(row_count, col_count))
    return row_count, col_count


def get_captcha_query(captcha_iframe):
    text_xpath = '//*[@id="rc-imageselect"]/div[2]/div[1]/div[1]/div[1]/strong'
    if captcha_iframe.is_element_present_by_xpath(text_xpath, wait_time=3):
        captcha_text = captcha_iframe.find_by_xpath(text_xpath).first['innerHTML']
        return captcha_text


def pick_checkboxes_matching_query(image_checkboxes, predicted_word_labels, query):
    matching_labels = []
    for i, image_labels in enumerate(predicted_word_labels):
        for label in image_labels:
            if " " in label:
                for word in label.split(" "):
                    if word == query:
                        matching_labels.append(i)
            elif label == query:
                matching_labels.append(i)

    matching_image_checkboxes = pick_checkboxes_from_positions(matching_labels, image_checkboxes)
    return matching_image_checkboxes


def click_checkboxes(checkboxes):
    if checkboxes:
        for checkbox in checkboxes:
            if checkbox['checkbox'].visible:
                checkbox['checkbox'].click()


def reload(captcha_iframe, total_guesses, correct_score):
    print("Reloading captcha iframe...")
    if captcha_iframe.is_element_present_by_id('recaptcha-reload-button', wait_time=3):
        recaptcha_reload_button = captcha_iframe.find_by_id('recaptcha-reload-button')
        recaptcha_reload_button.first.click()
        current_state = {'total_guesses': total_guesses+1, 'correct_score': correct_score}
        with open('logs/current_state.json','w+') as f:
            f.write(json.dumps(current_state))


def click_initial_checkbox(browser):
    if browser.is_element_present_by_name('undefined', wait_time=3):
        with browser.get_iframe('undefined') as iframe:
            print("Clicking initial checkbox.")
            if iframe.is_element_present_by_xpath('//div[@class="recaptcha-checkbox-checkmark"]', wait_time=3):
                captcha_checkbox = iframe.find_by_xpath('//div[@class="recaptcha-checkbox-checkmark"]')
                captcha_checkbox.first.click()
            else:
                print("Can't find initial checkbox.")
                browser.reload()
                start_guessing(browser)
    else:
        print("Can't find initial iframe.")
        browser.reload()
        start_guessing(browser)


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
        print("before clicking: ", browser.find_by_css('body > div').first.visible)
        click_initial_checkbox(browser)

        if browser.is_element_present_by_css('body > div > div:nth-child(4) > iframe', wait_time=3):
            print("Captcha iframe is present")
            guess_captcha(browser, neural_net)

    except StaleElementReferenceException:
        print("stale element exception, reloading")
        browser.reload()
        start_guessing(browser)
    except Exception as e:
        browser.reload()
        start_guessing(browser)

with splinter.Browser() as browser:
    url = "https://nocturnaltortoise.github.io/captcha"
    browser.visit(url)
    neural_net = nn.NeuralNetwork('weights/xception-less-data-weights.h5')
    start_guessing(browser)
