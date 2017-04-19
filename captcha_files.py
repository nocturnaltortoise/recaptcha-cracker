import glob
import json
import os

def delete_old_images():
    old_captcha_images = glob.glob('*captcha-*.jpg')
    for image in old_captcha_images:
        os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), image))

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
