class Captcha:
    def __init__(self, rows=0, cols=0, urls=[], checkboxes=[], query=''):
        self.rows = rows
        self.cols = cols
        self.image_url = ''
        self.checkboxes = checkboxes
        self.query = query

    def __str__(self):
        return "query: {0}, rows: {1}, cols: {2}, image_url: {3}, checkboxes: {4}".format(self.query, self.rows, self.cols, self.image_url, len(self.checkboxes))

class Checkbox:
    def __init__(self, position, element, image_path=None):
        self.position = position
        self.element = element
        self.image_path = image_path
        self.image_url = ''
