from captcha_interaction import CaptchaElement

class Captcha:
    def __init__(self, rows, cols, urls, checkboxes, query):
        self.rows = rows
        self.cols = cols
        self.image_url = ''
        self.changed_urls = []
        self.checkboxes = checkboxes
        self.query = query

class Checkbox:
    def __init__(self, position, element, image_path=None):
        self.position = position
        self.element = element
        self.image_path = image_path
