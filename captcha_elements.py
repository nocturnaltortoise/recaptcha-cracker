class Captcha:
    def __init__(self):
        self.rows = 0
        self.cols = 0
        self.image_url = ''
        self.checkboxes = []
        self.query = ''

    def __str__(self):
        return "query: {0}, rows: {1}, cols: {2}, image_url: {3}, checkboxes: {4}".format(self.query, self.rows, self.cols, self.image_url, len(self.checkboxes))

class Checkbox:
    def __init__(self, position, element, image_url):
        self.position = position
        self.element = element
        self.image_url = image_url
        self.image_path = None
        self.permanent_path = None
        self.predictions = []
