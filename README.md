# recaptcha-cracker
A program to automatically crack the challenges of Google's reCAPTCHA.

## Installing
- If you have Pipenv installed, there is an included Pipfile, so you can create a Python3 environment using ```pipenv --three```, and then you can run ```pipenv install```
- If you don't have Pipenv, then consult the Pipfile and Pipfile.lock for which packages to install using Pip (also consider installing Pipenv it's great: https://github.com/kennethreitz/pipenv/)
- You can either use the existing labels, categories and weights files or provide your own in the same format. If you do the latter, make sure to change config.py. If you wish to train a new network using this code, you will also need to change config.py and supply a suitable dataset.
- In order for Splinter to work, you will need geckodriver, which is the web driver behind Firefox, the default browser for Selenium. 
- There is a possibility for issues on Windows due to the lack of jpeg support in Pillow on Windows without libjpeg, which is difficult to install on Windows. This page might help: https://stackoverflow.com/questions/20672530/installing-pillow-on-windows-fails-to-find-libjpeg

## Running
- Once you have installed dependencies, you can run the CAPTCHA cracker by changing to the captcha-cracker directory and using ```python3 captcha_input.py``` (Assuming your python 3 install is ran by running ```python3```)
- To view CAPTCHAs previously seen by the CAPTCHA cracker, change directory to captcha-predictions-viewer and run ```python3 main.py``` 
- To retrain the network, instantiate a NeuralNetwork with a learning and decay rate, or uncomment the example at the bottom of nn.py and run nn.py
