import glob
import json
import os

def update_state_file(state_file_path, correct):
    with open(state_file_path, 'w+') as current_state_file:
        current_state = json.loads(current_state_file.read())
        correct_score = current_state['correct_score']
        total_guesses = current_state['total_guesses']
        if correct:
            new_state = {'total_guesses': total_guesses, 'correct_score': correct_score+1}
        else:
            new_state = {'total_guesses': total_guesses+1, 'correct_score': correct_score}
        current_state_file.write(json.dumps(new_state))

def delete_old_images():
    old_captcha_images = glob.glob('*captcha-*.jpg')
    for image in old_captcha_images:
        os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), image))
