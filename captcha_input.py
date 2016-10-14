from splinter import Browser
import random
import time


def guess_captcha(browser):
    image_iframe = browser.find_by_css('body > div > div:nth-child(4) > iframe')

    with browser.get_iframe(image_iframe.first['name']) as captcha_iframe, open('possible_categories.txt', 'a') as f:
        f.write('New Run. Categories asked for in CAPTCHAs written below.\n')
        correct_score = 0
        total_guesses = 0 # not necessarily separate captchas, one captcha with new images added would count as two
        # image_checkboxes = captcha_iframe.find_by_xpath('//div[@class="rc-imageselect-checkbox"]')

        categories = {}
        while True:

            rows = captcha_iframe.find_by_xpath('//*[@id="rc-imageselect-target"]/table/tbody/child::tr')
            cols = captcha_iframe.find_by_xpath('//*[@id="rc-imageselect-target"]/table/tbody/tr[1]/child::td')
            print("row count: ", len(rows))
            print("col count: ", len(cols))

            # need to keep getting images and image urls until this batch of image urls is the same as the last run
            # i.e. keep selecting images until the captcha stops replacing images

            image_checkboxes = []
            for i in range(1, len(rows)+1): # these numbers should be calculated by how big the grid is for the captcha
                for j in range(1, len(cols)+1):
                    checkbox_xpath = '//*[@id="rc-imageselect-target"]/table/tbody/tr[{0}]/td[{1}]/div'.format(i, j)
                    image_xpath = '//*[@id="rc-imageselect-target"]/table/tbody/tr[{0}]/td[{1}]/div/div[1]/img'.format(i, j)
                    image_urls = [image['src'] for image in captcha_iframe.find_by_xpath(image_xpath)]
                    if captcha_iframe.is_element_present_by_xpath(checkbox_xpath):
                        image_checkboxes += captcha_iframe.find_by_xpath(checkbox_xpath)
                    else:
                        recaptcha_reload_button = captcha_iframe.find_by_id('recaptcha-reload-button')
                        recaptcha_reload_button.click()
                        continue

            checkbox_num = random.randint(1,5)

            # get a set of random positions so we know which ones we picked
            random_positions = random.sample(range(len(image_checkboxes)), checkbox_num)

            random_checkboxes = []
            for pos in random_positions: # use these random positions to pick checkboxes (i.e. images)
                random_checkboxes.append(image_checkboxes[pos])

            # random_checkboxes = random.sample(image_checkboxes, checkbox_num)
            for checkbox in random_checkboxes:
                checkbox.click()

            # store attributes of td images, if they change after these clicks,
            # we pick some more to click from those

            # once we have a more intelligent method than random clicking,
            # we should examine the checkboxes we previously clicked,
            # and only examine those to determine which to click,
            # rather than testing all the images again

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


def record_captcha_question(captcha_iframe, f, categories):
    captcha_text = captcha_iframe.find_by_xpath('//*[@id="rc-imageselect"]/div[2]/div[1]/div[1]/div[1]/strong').first['innerHTML']

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
