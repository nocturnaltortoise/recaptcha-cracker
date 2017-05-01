# recaptcha-cracker
A program to automatically crack the challenges of Google's reCAPTCHA.

## Installing
- If you have Pipenv installed, there is an included Pipfile, so you can run ```pipenv install```
- If you don't have Pipenv, then consult the Pipfile for which packages to install using Pip
- You can either use the existing labels, categories and weights files or provide your own in the same format. If you do the latter, make sure to change config.py. If you wish to train a new network using this code, you will also need to change config.py and supply a suitable dataset.

## Running
- Once you have installed dependencies, you can run the CAPTCHA cracker by changing to the captcha-cracker directory and using ```python3 captcha_input.py``` (Assuming your python 3 install is ran by running ```python3```)
- To view CAPTCHAs previously seen by the CAPTCHA cracker, change directory to captcha-predictions-viewer and run ```python3 main.py``` 
