from splinter import Browser
import random
import time
import math
import requests
from PIL import Image
import re


def find_image_url(captcha_iframe, picked_positions=None, row_count=None):
    if picked_positions:
        image_urls = []
        for position, i in enumerate(picked_positions):
            # turn 1d array index into 2d position
            x = position // row_count
            y = position % row_count

            # make sure counting starts from 1 not 0
            if x == 0:
                x += 1
            if y == 0:
                y += 1

            changed_image_xpath = '//*[@id="rc-imageselect-target"]/table/tbody/tr[{0}]/td[{1}]/div/div[1]/img'\
                .format(x, y)
            if captcha_iframe.is_element_present_by_xpath(changed_image_xpath, wait_time=3):
                image_url = captcha_iframe.find_by_xpath(changed_image_xpath)['src']
                image_urls.append(image_url)
    else:
        image_xpath = '//*[@id="rc-imageselect-target"]/table/tbody/tr[1]/td[1]/div/div[1]/img'
        if captcha_iframe.is_element_present_by_xpath(image_xpath, wait_time=3):
            image_url = captcha_iframe.find_by_xpath(image_xpath)['src']

    return image_url


def pick_checkboxes_from_positions(random_positions, image_checkboxes):
    random_checkboxes = []
    for pos in random_positions:  # use these random positions to pick checkboxes (i.e. images)
        random_checkboxes.append(image_checkboxes[pos])

    # print(len(random_checkboxes))
    return random_checkboxes


def pick_random_checkboxes(image_checkboxes):
    random_positions = None
    if len(image_checkboxes) != 1:
        checkbox_num = random.randint(1, math.ceil(len(image_checkboxes)/2))

        # get a set of random positions so we know which ones we picked
        random_positions = random.sample(range(len(image_checkboxes)), checkbox_num)

        random_checkboxes = pick_checkboxes_from_positions(random_positions, image_checkboxes)

        # random_checkboxes = random.sample(image_checkboxes, checkbox_num)
        for checkbox in random_checkboxes:
            checkbox.click()
    else:
        print("image checkbox length:", len(image_checkboxes))
    return random_positions


def get_image_checkboxes(rows, cols, captcha_iframe):
    image_checkboxes = []
    dimensions_divs = []
    for i in range(1, len(rows)+1):  # these numbers should be calculated by how big the grid is for the captcha
        for j in range(1, len(cols)+1):
            checkbox_xpath = '//*[@id="rc-imageselect-target"]/table/tbody/tr[{0}]/td[{1}]/div'.format(i, j)
            dimensions_div_xpath = '//*[@id="rc-imageselect-target"]/table/tbody/tr[{0}]/td[{1}]/div/div'.format(i, j)

            if captcha_iframe.is_element_present_by_xpath(checkbox_xpath):
                image_checkboxes += captcha_iframe.find_by_xpath(checkbox_xpath)

            if captcha_iframe.is_element_present_by_xpath(dimensions_div_xpath):
                dimensions_divs += captcha_iframe.find_by_xpath(dimensions_div_xpath)

    return image_checkboxes, dimensions_divs


def verify(captcha_iframe, f, categories, correct_score, total_guesses):
    verify_button = captcha_iframe.find_by_id('recaptcha-verify-button')
    record_captcha_question(captcha_iframe, f, categories)
    time.sleep(2)
    verify_button.first.click()

    not_select_all_images_error = captcha_iframe.is_text_not_present('Please select all matching images.')
    not_retry_error = captcha_iframe.is_text_not_present('Please try again.')
    not_select_more_images_error = captcha_iframe.is_text_not_present('Please also check the new images.')
    if not_select_all_images_error and not_retry_error and not_select_more_images_error:
        correct_score += 1
    total_guesses += 1
    print("Total possibly correct: {correct}".format(correct=correct_score))
    print("Total guesses: {guesses}".format(guesses=total_guesses))
    print("Percentage: {percent}".format(percent=float(correct_score)/total_guesses))


def guess_captcha(browser):
    image_iframe = browser.find_by_css('body > div > div:nth-child(4) > iframe')

    with browser.get_iframe(image_iframe.first['name']) as captcha_iframe, open('possible_categories.txt', 'a') as f:
        f.write('New Run. Categories asked for in CAPTCHAs written below.\n')
        correct_score = 0
        total_guesses = 0  # not necessarily separate captchas, one captcha with new images added would count as two
        # image_checkboxes = captcha_iframe.find_by_xpath('//div[@class="rc-imageselect-checkbox"]')

        categories = {}
        new_run = True
        while True:
            rows = captcha_iframe.find_by_xpath('//*[@id="rc-imageselect-target"]/table/tbody/child::tr')
            cols = captcha_iframe.find_by_xpath('//*[@id="rc-imageselect-target"]/table/tbody/tr[1]/child::td')
            row_count = len(rows)
            col_count = len(cols)
            print("row count: ", row_count)
            print("col count: ", col_count)

            # need to keep getting images and image urls until this batch of image urls is the same as the last run
            # i.e. keep selecting images until the captcha stops replacing images
            checkbox_xpath = '//*[@id="rc-imageselect-target"]/table/tbody/tr[1]/td[1]/div'

            if captcha_iframe.is_element_not_present_by_xpath(checkbox_xpath):
                recaptcha_reload_button = captcha_iframe.find_by_id('recaptcha-reload-button')
                recaptcha_reload_button.click()
                continue

            if new_run:
                image_checkboxes, dimensions_divs = get_image_checkboxes(rows, cols, captcha_iframe)

                image_dimensions = []
                dimension_number_regex = re.compile('[0-9]+')
                for div in dimensions_divs:
                    if div['style'] is not '':
                        print(div['style'].split(';')[0:2])
                        width, height = div['style'].split(';')[0:2]
                        x = float(re.search(dimension_number_regex, width).group(0))
                        y = float(re.search(dimension_number_regex, height).group(0))
                        image_dimensions.append((x, y))

                image_url = find_image_url(captcha_iframe)

                img = Image.open(requests.get(image_url, stream=True).raw)
                img.save("test.jpg", "JPEG")

                for row in range(row_count):
                    for col in range(col_count):
                        width, height = image_dimensions[0]
                        dimensions = (col * width, row * height, col * width + width, row * height + height)
                        print(dimensions)
                        individual_captcha_image = img.crop(dimensions)
                        individual_captcha_image.save("captcha-{0}-{1}.jpg".format(row, col), "JPEG")
                # there's a difference between the div size and the size of the parts of the image
                # size of the parts of the image are smaller

                # getting element is no longer attached to DOM errors with this?

                # split into several images based off td sizes?
                # feed images to inception - later we may wish to at least train inception with different datasets
                # get best guess back
                # pick images with best guesses close to question

                print("image url: ", image_url)
                picked_positions = pick_random_checkboxes(image_checkboxes)
                print("picked these positions:", picked_positions)
                new_image_urls = find_image_url(captcha_iframe, picked_positions, row_count=row_count)
                print("new image url: ", new_image_urls)

                # store attributes of td images, if they change after these clicks,
                # we pick some more to click from those
                new_run = False

            if any(image_url != new_image_url for new_image_url in new_image_urls):
                # comparing the urls for images seems to work in terms of detecting a change

                print("set of images has changed")
                # refresh the image checkboxes because images may have disappeared/appeared

                # if not new_run:
                time.sleep(1)
                new_image_checkboxes, new_dimensions_divs = get_image_checkboxes(rows, cols, captcha_iframe)

                picked_positions = pick_random_checkboxes(pick_checkboxes_from_positions(picked_positions,
                                                                                         new_image_checkboxes))
                print("picked these positions after click:", picked_positions)
                if not picked_positions:
                    verify(captcha_iframe, f, categories, correct_score, total_guesses)
                    new_run = True
                image_url = find_image_url(captcha_iframe)
                new_image_urls = find_image_url(captcha_iframe, picked_positions, row_count=row_count)
            else:
                verify(captcha_iframe, f, categories, correct_score, total_guesses)
                new_run = True


def record_captcha_question(captcha_iframe, f, categories):
    iframe_xpath = '//*[@id="rc-imageselect"]/div[2]/div[1]/div[1]/div[1]/strong'
    captcha_text = captcha_iframe.find_by_xpath(iframe_xpath).first['innerHTML']

    print("categories before:", categories)
    if captcha_text in categories:
        categories[captcha_text] += 1
    else:
        categories[captcha_text] = 1
        f.write(captcha_text)
        f.write('\n')
    print("categories after:", categories)

with Browser() as browser:
    url = "https://nocturnaltortoise.github.io/captcha"
    browser.visit(url)

    try:
        with browser.get_iframe('undefined') as iframe:
            captcha_checkbox = iframe.find_by_xpath('//div[@class="recaptcha-checkbox-checkmark"]')
            captcha_checkbox.first.click()

        if browser.is_element_present_by_css('body > div > div:nth-child(4) > iframe', wait_time=10):
            guess_captcha(browser)

    except Exception as e:
        print(e)
        browser.screenshot('error')
