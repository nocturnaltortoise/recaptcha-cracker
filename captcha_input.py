from splinter import Browser
import random
import time

def guess_captcha(browser):
    image_iframe = browser.find_by_css('body > div > div:nth-child(4) > iframe')

    with browser.get_iframe(image_iframe.first['name']) as captcha_iframe, open('possible_categories.txt', 'a') as f:
        correct_score = 0
        total_guesses = 0 # not necessarily separate captchas, one captcha with new images added would count as two
        # image_checkboxes = captcha_iframe.find_by_xpath('//div[@class="rc-imageselect-checkbox"]')

        categories = {}
        while True:
            image_checkboxes = []
            for i in range(1,4):
                for j in range(1,4):
                    image_checkboxes += captcha_iframe.find_by_xpath('//*[@id="rc-imageselect-target"]/table/tbody/tr[{0}]/td[{1}]/div'.format(i, j))

            checkbox_num = random.randint(1,5)
            random_checkboxes = random.sample(image_checkboxes, checkbox_num)
            for checkbox in random_checkboxes:
                checkbox.click()

            # verify_button = iframe.find_by_id('recaptcha-verify-button')
            verify_button = captcha_iframe.find_by_id('recaptcha-verify-button')
            # verify_button = iframe.find_by_xpath('//div[@class="rc-button-default"]')
            record_captcha_question(captcha_iframe, f, categories)
            time.sleep(2)
            verify_button.first.click()

            not_select_all_images_error = captcha_iframe.is_text_not_present('Please select all matching images.')
            not_retry_error = captcha_iframe.is_text_not_present('Please try again.')
            not_select_more_images_error = captcha_iframe.is_text_not_present('Please also check the new images.')
            if not_select_all_images_error and not_retry_error and not_select_more_images_error:
                correct_score += 1
            total_guesses += 1
            print("Total correct (probably not correct): {correct}".format(correct=correct_score))
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
