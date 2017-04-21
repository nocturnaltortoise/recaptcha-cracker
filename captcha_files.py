import glob
import json
import os

def delete_old_images():
    old_captcha_images = glob.glob('*captcha-*.jpg')
    for image in old_captcha_images:
        os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), image))

def write_guesses_to_file(captcha, matching_checkboxes, correct):
    captcha_text = captcha.query
    folder = captcha.random_id
    rows = captcha.rows
    cols = captcha.cols
    checkboxes = [{"position": checkbox.position,
                   "path": checkbox.permanent_path,
                   "predictions": checkbox.predictions,
                   "matching": checkbox in matching_checkboxes} for checkbox in captcha.checkboxes]
    # predictions = [checkbox.predictions for checkbox in captcha.checkboxes]
    # matching_predictions = [checkbox.predictions for checkbox in matching_checkboxes]
    if os.path.exists('guesses.json'):
        with open('guesses.json', 'r') as guess_file:
            existing_predictions = guess_file.read()
            json_predictions = json.loads(existing_predictions)
    else:
        json_predictions = {}

    with open('guesses.json', 'w+') as guess_file:
        if captcha_text not in json_predictions:
            json_predictions[captcha_text] = {}
        json_predictions[captcha_text][folder] = {
            "rows": rows,
            "cols": cols,
            "checkboxes": checkboxes,
            "correct": correct
        }
        guess_file.write(json.dumps(json_predictions))
